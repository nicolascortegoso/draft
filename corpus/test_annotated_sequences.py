import json
import unittest
from glob import glob


class TestYakutAnalyzer(unittest.TestCase):

    base_path = '/Users/nicolas/projects/yakutmorph/corpus'
    index_path = f'{base_path}/data_index.csv'

    def get_file_paths(self, folder, subfolder):
        file_paths = sorted(glob(f'{self.base_path}/{folder}/{subfolder}/*.json'))
        return file_paths

    def read_indexed_folders(self):
        folders = dict()
        with open(self.index_path, 'r') as file:
            lines = file.readlines()[1:]
            for line in lines:
                folder, content, reference = line.split('\t')
                folders[folder] = {'content': content, 'reference': reference}
        return folders

    def get_filename(self, file_path):
        return file_path.split('/')[-1]

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for sequence in data:
                    yield sequence
            except Exception as e:
                print(f'Error {e} in file {file_path}')


    def setUp(self):
        self.data = self.read_indexed_folders()
        for folder in self.data:
            self.data[folder].update({
                'source': {
                    self.get_filename(file_path): self.read_file(file_path)
                    for file_path in self.get_file_paths(folder, 'source')
                },
                'processed': {
                    self.get_filename(file_path): self.read_file(file_path)
                    for file_path in self.get_file_paths(folder, 'processed')
                }
            })

    def update_statistics(self, data, filename):
        file_path = f'{self.base_path}/statistics/{filename}.json'
        with open(file_path, 'w', encoding='utf8') as json_file:
            json.dump(data, json_file, indent=3, ensure_ascii=False)

    def update_markdown_file(self, filename, mappings):
        source_template = f'{self.base_path}/test/templates/{filename}.stub'
        target_template = f'{self.base_path}/{filename}.md'
        filled_template = self.fill_template(source_template, mappings)
        with open(target_template, 'w') as f:
            f.write(filled_template)

    def fill_template(self, template_file, mappings):
        with open(template_file) as f:
            template = f.read()
        for key, value in mappings.items():
            template = template.replace(key, value)
        return template

    def construct_table(self, columns, rows, foot=None):
        table_content = list()
        header = '|' + '|'.join(columns) + '|'
        line_separator = '|:---' * len(columns) + '|'
        table_content.append(header)
        table_content.append(line_separator)
        for row in rows:
            formated_row = ' | ' + ' | '.join(row) + ' | '
            table_content.append(formated_row)
        if foot:
            formated_foot = ' | ' + ' | '.join(foot) + ' | '
            table_content.append(formated_foot)
        return '\n'.join(table_content)

    # def test_empty_string(self):
    #     for folder, content in self.data.items():
    #         for file, sequences in content['source'].items():
    #             for i, sequence in enumerate(sequences):
    #                 sah = sequence['sah']
    #                 assert len(sah) != 0, f"Empty string found in folder '{folder}', file '{file}', sequence n: {i}"

    def test_sequences(self):
        wordforms = dict()
        roots = dict()
        affixes = dict()

        for folder, content in self.data.items():
            for file, sequences in content['processed'].items():
                for i, sequence in enumerate(sequences):
                    expected_pos = 1
                    analyses = sequence['parses']
                    for token in analyses:
                        curr_pos = token['pos']
                        assert curr_pos == expected_pos, f"Incorrect token sequence in '{folder}', file '{file}', sequence id: {sequence['id']}"
                        expected_pos += 1
                        root = token.get('root', None)
                        if root:
                            wordform = token['text'].lower()
                            wordforms[wordform] = wordforms.get(wordform, 0) + 1
                            affixes_list = token['affixes']
                            root_type = affixes_list[0]
                            roots[(root, root_type)] = roots.get((root, root_type), 0) + 1
                            for affix in affixes_list[1:]:
                                affixes[affix] = affixes.get(affix, 0) + 1

        wordform_columns = ('word form', 'frequency')
        sorted_wordforms = dict(sorted(wordforms.items(), key=lambda item: item[1], reverse=True))
        wordform_rows = ((wordform, str(frequency)) for wordform, frequency in sorted_wordforms.items())
        wordforms_table = self.construct_table(wordform_columns, wordform_rows)
        self.update_markdown_file('wordform_distribution', {
            '{TOTAL_WORDFORMS}': str(sum(sorted_wordforms.values())),
            '{TOTAL_UNIQUE_WORDFORMS}': str(len(set(sorted_wordforms.keys()))),
            '{TABLE}': wordforms_table
        })
        self.update_statistics(sorted_wordforms, 'wordform_distribution')

        root_columns = ('root', 'type', 'frequency')
        sorted_roots = dict(sorted(roots.items(), key=lambda item: item[1], reverse=True))
        root_rows = ((root, type, str(frequency)) for (root, type), frequency in sorted_roots.items())
        roots_table = self.construct_table(root_columns, root_rows)
        self.update_markdown_file('root_distribution', {
            '{TOTAL_ROOTS}': str(sum(sorted_roots.values())),
            '{TOTAL_UNIQUE_ROOTS}': str(len(set(sorted_roots.keys()))),
            '{TABLE}': roots_table
        })
        formated_sorted_roots = {f'{k[0]}{k[1]}': v for k, v in sorted_roots.items()}
        self.update_statistics(formated_sorted_roots, 'root_distribution')

        affix_columns = ('affix', 'frequency')
        sorted_affixes = dict(sorted(affixes.items(), key=lambda item: item[1], reverse=True))
        affix_rows = ((affix, str(frequency)) for affix, frequency in sorted_affixes.items())
        affixes_table = self.construct_table(affix_columns, affix_rows)
        self.update_markdown_file('affix_distribution', {
            '{TOTAL_AFFIXES}': str(sum(sorted_affixes.values())),
            '{TOTAL_UNIQUE_AFFIXES}': str(len(set(sorted_affixes.keys()))),
            '{TABLE}': affixes_table
        })
        self.update_statistics(sorted_affixes, 'affix_distribution')

    
    def test_chapters(self):

        chapters = dict()

        for folder, content in self.data.items():
            chapters[folder] = dict()
            for file, sequences in content['source'].items():
                file_chapters = [sequence['chapter'] for sequence in sequences]
                self.assertEqual(
                    len(set(file_chapters)), 1, msg={
                        'Folder': folder,
                        'file': file,
                        'chapters': file_chapters
                    }
                )
                chapter_title = list(file_chapters)[0]
                if chapter_title not in chapters[folder]:
                    chapters[folder][chapter_title] = 0
                chapters[folder][chapter_title] += len(file_chapters)
        
        columns = ('folder', 'chapter', 'sequences')
        rows = list()
        for folder, chapter_title in chapters.items():
            for key, value in chapter_title.items():
                row = (f'{folder}', key, f'{value}')
                rows.append(row)
        chapters_table = self.construct_table(columns, rows)

        self.update_markdown_file('annotated_chapters', {
            '{TABLE}': chapters_table
        })

    
    # def test_topics(self):

    #     for folder, content in self.data.items():
    #         for file, sequences in content['source'].items():
    #             topics = set(
    #                 [sequence['topic'] for sequence in sequences]
    #             )
    #             self.assertEqual(
    #                 len(topics), 1, msg={
    #                     'Folder': folder,
    #                     'file': file,
    #                     'topics': topics
    #                 }
    #             )

    def test_progress(self):

        columns = ('Folder', 'Content', 'Total sequences', 'Processed sequences', 'Annotation progress', 'Source')
        rows = list()

        total_source = 0
        total_processed = 0

        for folder, content in self.data.items():
            n_source = 0
            n_processed = 0
            for file, sequences in content['source'].items():
                n_source += len(list(sequences))
            for file, sequences in content['processed'].items():
                n_processed += len(list(sequences))
            total_source += n_source
            total_processed += n_processed

            annotation_progress = f'{(n_processed / n_source) * 100:.2f}%'
            row_elements = [folder, content['content'], n_source, n_processed, annotation_progress, content['reference']]
            row = [str(i) for i in row_elements]
            rows.append(row)

        total_progress = f"{(total_processed / total_source) * 100:.2f}%"
        foot = ['', 'TOTAL', total_source, total_processed, total_progress, '']
        foot = [str(i) for i in foot]
        progress_table = self.construct_table(columns, rows, foot)

        self.update_markdown_file('annotation_progress', {
            '{PROGRESS_TABLE}': progress_table,
        })


if __name__ == '__main__':
    unittest.main()
