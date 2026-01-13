import json
from glob import glob
import unittest



class TestYakutAnalyzer(unittest.TestCase):
    def setUp(self):
        self.index_path = '/Users/nicolas/projects/yakutmorph/corpus/data_index.csv'

    def read_file(self, file_path):
        with open(file_path, 'r') as f:
            sequences = json.load(f)
            return sequences

    def surface_analysis(self, row):
        surface = row['text'].lower().strip()
        root = row.get('root', None)
        if root:
            suffixes = ''.join(row['affixes'])
            analysis = f'{root}{suffixes}'.strip()
            return surface, '', '', '', analysis, str(row['pos'])
        return False

    def test_corpus(self):
        with open('/Users/nicolas/projects/yakutmorph/tests/analyses/annotated_corpus.csv', 'w') as wf:
            header = '\t'.join(['surface', 'stem', 'surface_gloss', 'stem_gloss', 'analysis', 'page', 'section', 'notes', 'source'])
            wf.write(f"{header}\n")
            with open(self.index_path, 'r') as file:
                lines = file.readlines()[1:]
                for line in lines:
                    folder, content, source = line.split('\t')
                    processed_files = glob(f'/Users/nicolas/projects/yakutmorph/corpus/{folder}/processed/*.json')
                    for file_path in processed_files:
                        sequences = self.read_file(file_path)
                        for seq in sequences:
                            file_name = '/'.join(file_path.split('/')[-3:])
                            reference = str(seq['id']), seq['sah'], file_name
                            for token in seq['parses']:
                                content = self.surface_analysis(token)
                                if content:
                                    row = '\t'.join(content + reference)
                                    wf.write(f'{row}\n')




if __name__ == '__main__':
    unittest.main()