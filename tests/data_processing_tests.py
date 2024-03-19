import unittest
from data_preprocessors.raw_processor import RawDataProcessor

class TestProcessing(unittest.TestCase):

   def test_sources_targets_same_length(self):

        rdp = RawDataProcessor()
        sources, targets = rdp.join_source_target()
        
        self.assertEqual(len(sources), len(targets))

if __name__ == '__main__':
    unittest.main()