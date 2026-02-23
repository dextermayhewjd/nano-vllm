import unittest

from nanovllm.engine.sequence import Sequence, SequenceStatus


class SequenceTests(unittest.TestCase):
    def test_init_and_append_flow(self):
        seq = Sequence([1, 2, 3], seq_id=7)

        self.assertEqual(seq.seq_id, 7)
        self.assertEqual(seq.status, SequenceStatus.WAITING)
        self.assertEqual(seq.num_tokens, 3)
        self.assertEqual(seq.num_completion_tokens, 0)

        seq.append_token(4)
        seq.append_token(5)

        self.assertEqual(seq.prompt_token_ids, [1, 2, 3])
        self.assertEqual(seq.completion_token_ids, [4, 5])
        self.assertEqual(seq.num_completion_tokens, 2)

    def test_empty_token_ids_raises(self):
        with self.assertRaises(ValueError):
            Sequence([])


if __name__ == "__main__":
    unittest.main()
