"""CPU tensors with real pinned DataProto; fake workers do not validate GPUs."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

DEPENDENCIES = ("torch", "numpy", "tensordict", "transformers", "requests")
AVAILABLE = all(importlib.util.find_spec(name) is not None for name in DEPENDENCIES)
if AVAILABLE:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/Search-R1"))
    import numpy as np
    import torch
    from verl import DataProto
    from verl.protocol import unpad_dataproto
    from procurement_rl.training.batching import pad_for_workers
    from procurement_rl.training.generation import ProcurementGenerationManager, _pad, _positions
    from procurement_rl.training.scoring import score_information_gains


class Config(dict):
    __getattr__ = dict.__getitem__


class Tokenizer:
    pad_token_id, eos_token_id = 0, 1

    def encode(self, text, add_special_tokens=False):
        return [ord(character) + 2 for character in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(value) - 2) for value in ids if int(value) > 1)


@unittest.skipUnless(AVAILABLE, "CPU tensor adapter tests require the Search-R1 environment")
class TensorAdapterTests(unittest.TestCase):
    def test_optimizer_records_real_change_and_rejects_nonfinite_gradient(self):
        from procurement_rl.training.workers import AuditedPPOActor
        actor = object.__new__(AuditedPPOActor)
        actor.config = SimpleNamespace(grad_clip=1.0)
        actor.actor_module = torch.nn.Linear(2, 1, bias=False)
        actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.1)
        actor.sampled_deltas = []
        parameter = next(actor.actor_module.parameters())
        parameter.grad = torch.ones_like(parameter)
        actor._optimizer_step()
        self.assertGreater(actor.sampled_deltas[0], 0)
        before = parameter.detach().clone()
        parameter.grad = torch.full_like(parameter, float('nan'))
        with self.assertRaises(FloatingPointError):
            actor._optimizer_step()
        self.assertTrue(torch.equal(parameter.detach(), before))

    def test_padding_one_active_row_on_three_workers(self):
        for size in (1, 2, 3, 4):
            data = DataProto.from_dict({"x": torch.arange(size).view(-1, 1)})
            padded, count = pad_for_workers(data, 3)
            self.assertEqual(len(padded) % 3, 0)
            self.assertTrue(torch.equal(unpad_dataproto(padded, count).batch["x"], data.batch["x"]))

    def _rollout(self):
        tokenizer = Tokenizer()
        calls = []
        class Worker:
            world_size = 3

            def generate_sequences(worker, data):
                self.assertEqual(len(data) % 3, 0)
                self.assertEqual(data.non_tensor_batch, {})
                calls.append(len(data))
                rows = []
                for ids, mask in zip(data.batch["input_ids"], data.batch["attention_mask"]):
                    prompt = tokenizer.decode(ids[mask.bool()].tolist())
                    message = ("<search>catalog</search>unused" if prompt == "P" else
                               "<evidence>D</evidence><answer>offer</answer>unused")
                    rows.append(tokenizer.encode(message))
                responses, rmask = _pad(rows, 0)
                return DataProto.from_dict({"responses": responses,
                    "attention_mask": torch.cat((data.batch["attention_mask"], rmask), -1)})
        config = SimpleNamespace(max_start_length=32, max_turns=2, max_response_length=128,
                                 max_obs_length=256, max_prompt_length=2048, num_gpus=3,
                                 topk=2, search_url="unused")
        rollout = Config(log_prob_micro_batch_size=3, log_prob_max_token_len_per_gpu=2048,
                         log_prob_use_dynamic_bsz=False, temperature=1.0)
        manager = ProcurementGenerationManager(tokenizer, Worker(), config, {}, rollout)
        manager._retrieve = lambda queries: [[{"id": "D", "contents": "public catalog"}] for _ in queries]
        ids, mask = _pad([tokenizer.encode("P"), tokenizer.encode("Q")], 0, left=True)
        data = DataProto.from_dict({"input_ids": ids, "attention_mask": mask, "position_ids": _positions(mask)})
        output = manager.run_llm_loop(data, None)
        return output, tokenizer, calls

    def test_actual_tokens_provenance_and_shrinking_active_batch(self):
        output, tokenizer, calls = self._rollout()
        self.assertEqual(calls, [3, 3])
        traces = output.non_tensor_batch["procurement_trace"]
        self.assertEqual([len(t["observation_ends"]) for t in traces], [1, 0])
        trace = traces[0]
        self.assertEqual(trace["observations"][0]["query"], "catalog")
        self.assertEqual(trace["observations"][0]["documents"][0]["id"], "D")
        self.assertNotIn("unused", "".join(trace["assistant_messages"]))
        expected = (tokenizer.encode(trace["assistant_messages"][0]) +
                    tokenizer.encode(trace["observations"][0]["text"]) +
                    tokenizer.encode(trace["assistant_messages"][1]))
        self.assertEqual(expected, trace["response_token_ids"])
        width = output.batch["responses"].shape[-1]
        self.assertTrue(torch.equal(output.batch["info_mask"][:, -width:],
                                    (output.batch["turn_ids"] >= 0).long()))
        self.assertGreater(trace["generated_tokens"], sum(len(tokenizer.encode(t)) for t in trace["assistant_messages"]))

    def test_teacher_forcing_deduplicates_initial_state_and_isolates_gold(self):
        output, tokenizer, _ = self._rollout()
        traces = list(output.non_tensor_batch["procurement_trace"])
        traces[1] = copy.deepcopy(traces[0])
        output.non_tensor_batch["procurement_trace"] = np.array(traces, dtype=object)
        output.non_tensor_batch["reward_model"] = np.array(
            [{"ground_truth": {"gold_offer_id": "GOLD"}}] * 2, dtype=object)
        original = output.batch["input_ids"].clone()
        seen = []
        class Worker:
            world_size = 3

            def compute_log_prob(worker, data):
                self.assertEqual(len(data) % 3, 0)
                self.assertEqual(data.non_tensor_batch, {})
                rows = []
                for ids, mask in zip(data.batch["prompts"], data.batch["attention_mask"]):
                    prompt = tokenizer.decode(ids[mask[:len(ids)].bool()])
                    self.assertNotIn("GOLD", prompt)
                    self.assertNotIn("<answer>offer", prompt)
                    seen.append(prompt)
                    rows.append([-2.0 if "<information>" in prompt else -4.0] * data.batch["responses"].shape[-1])
                return DataProto.from_dict({"old_log_probs": torch.tensor(rows)})
        config = Config(data=Config(max_prompt_length=2048), procurement={"gold_score_micro_batch_size": 1})
        gains, metrics = score_information_gains(output, Worker(), tokenizer, config)
        self.assertEqual(gains, [[2.0], [2.0]])
        self.assertEqual(metrics["ig/unique_scoring_states"], 2)
        self.assertEqual(metrics["ig/scored_states_with_repeats"], 3)
        self.assertGreater(metrics["ig/scored_tokens"], metrics["ig/unique_state_tokens"])
        self.assertGreaterEqual(metrics["ig/allocated_tokens_with_padding"], metrics["ig/scored_tokens"])
        self.assertEqual(len(seen), 3)  # two unique states plus worker padding
        self.assertTrue(torch.equal(output.batch["input_ids"], original))
        self.assertNotIn("GOLD", tokenizer.decode(original[0]))

    def test_driver_runs_exact_updates_and_final_checkpoint(self):
        if not all(importlib.util.find_spec(name) for name in ("ray", "omegaconf", "codetiming", "pandas")):
            self.skipTest("Driver integration needs the bundled training dependencies")
        from procurement_rl.training.trainer import ProcurementTrainer
        output, tokenizer, _ = self._rollout()
        trainer = object.__new__(ProcurementTrainer)
        trainer.tokenizer = tokenizer
        trainer.config = Config(
            trainer=Config(total_epochs=2, val_before_train=True, val_only=False, test_freq=2, save_freq=2),
            actor_rollout_ref=Config(rollout=Config(n_agent=2)),
            procurement=Config(mode="grpo", ig_weight=1.0, outcome_weight=1.0))
        trainer.total_training_steps = 5
        base = {"input_ids": torch.ones((1, 1), dtype=torch.long),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
                "position_ids": torch.zeros((1, 1), dtype=torch.long)}
        trainer.train_dataloader = [base] * 3
        trainer._generation_manager = lambda: None
        def rollout(batch, manager):
            batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
            return batch.union(copy.deepcopy(output))
        trainer._rollout = rollout
        trainer._balance_batch = lambda batch, metrics: None
        logs, validations, checkpoints, updates = [], [], [], []
        trainer._log = lambda metrics: logs.append((trainer.global_steps, metrics))
        def validate():
            validations.append(trainer.global_steps)
            return {"procurement/val/reward": 0.0}
        trainer._validate = validate
        trainer._save_checkpoint = lambda: checkpoints.append(trainer.global_steps)
        class Reward:
            pending_metrics = {"procurement/train/reward": 0.5}
            def __call__(reward, batch):
                values = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
                positions = torch.nonzero(batch.batch["turn_ids"][0] >= 0).flatten()
                values[0, positions[-1]] = 1.0
                return values
        class Worker:
            def compute_log_prob(worker, batch):
                self.assertEqual(batch.non_tensor_batch, {})
                return DataProto.from_dict({"old_log_probs": torch.zeros_like(batch.batch["responses"], dtype=torch.float32)})
            def compute_ref_log_prob(worker, batch):
                self.assertEqual(batch.non_tensor_batch, {})
                return DataProto.from_dict({"ref_log_prob": torch.zeros_like(batch.batch["responses"], dtype=torch.float32)})
            def update_actor(worker, batch):
                self.assertEqual(batch.non_tensor_batch, {})
                mask = batch.batch["loss_mask"].bool()
                self.assertTrue(torch.all(batch.batch["advantages"][~mask] == 0))
                self.assertTrue(torch.any(batch.batch["advantages"][mask] != 0))
                updates.append(trainer.global_steps)
                return SimpleNamespace(meta_info={"metrics": {"actor/pg_loss": [0.0]}})
        trainer.reward_fn = Reward()
        trainer.actor_rollout_wg = trainer.ref_policy_wg = Worker()
        trainer.fit()
        self.assertEqual(updates, [1, 2, 3, 4, 5])
        self.assertEqual(validations, [0, 2, 4, 5])
        self.assertEqual(checkpoints, [2, 4, 5])
        self.assertEqual(trainer.global_steps, 5)

    def test_validation_weights_partial_batch_by_examples(self):
        if not all(importlib.util.find_spec(name) for name in ("ray", "omegaconf", "codetiming", "pandas")):
            self.skipTest("Driver integration needs the bundled training dependencies")
        from procurement_rl.training.trainer import ProcurementTrainer
        trainer = object.__new__(ProcurementTrainer)
        output, _, _ = self._rollout()
        output.non_tensor_batch["data_source"] = np.array(["procurement"] * 2, dtype=object)
        def dictionary(batch):
            return {**dict(batch.batch.items()), **batch.non_tensor_batch}
        trainer.val_dataloader = [dictionary(output), dictionary(output[:1])]
        trainer._generation_manager = lambda validation: None
        trainer._rollout = lambda batch, manager: batch
        class Reward:
            pending_metrics = {}
            def __call__(reward, batch):
                value = 1.0 if len(batch) == 2 else 0.0
                reward.pending_metrics = {"procurement/val/reward": value}
                return torch.full((len(batch), 1), value)
        trainer.val_reward_fn = Reward()
        metrics = trainer._validate()
        self.assertAlmostEqual(metrics["procurement/val/reward"], 2 / 3)
        self.assertAlmostEqual(metrics["val/test_score/procurement"], 2 / 3)
        self.assertEqual(metrics["procurement/val/examples"], 3)


if __name__ == "__main__":
    unittest.main()
