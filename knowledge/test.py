from KnowledgeBase import KnowledgeBase
import unittest

class TestKB(unittest.TestCase):
    """Unit tests for the Knowledge Base Class
        Make sure all functions act as they should

        call via `$python knowledge/test.py` to ensure validity
    """
    def test_get_by_id(self):
        KB = KnowledgeBase()
        query = KB.get_entity_from_id('Q1225')
        self.assertEqual(query, 'Bruce Springsteen', 'Unable to find ID')

    def test_get_name_from_alias(self):
        KB = KnowledgeBase()
        query = KB.get_name_from_alias('Henry III')
        self.assertEqual(query, 'Henry III of England', 'Unable to get name from alias')
    
    def test_get_translation(self):
        KB = KnowledgeBase()
        query = KB.get_translation_from_name('Henry III of England', 'ko')
        self.assertEqual(query, '헨리 3세', 'Unable to get correct translation')

    def test_get(self):
        KB = KnowledgeBase()
        query = KB.get('Ulan-Bator', 'ko')
        self.assertEqual(query, '울란바토르')

    def test_none(self):
        KB = KnowledgeBase()
        query = KB.get('Not a real entitty', 'ja')
        self.assertIsNone(query)
    

if __name__ == "__main__":
    unittest.main()