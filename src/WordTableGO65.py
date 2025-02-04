import pickle
from collections import defaultdict, Counter
from tqdm import tqdm


class Predict(object):
    def __init__(self):
        self.function_label = ['ATP-dependent activity', 'Antioxidant activity', 'Binding',
                               'Purine ribonucleoside triphosphate binding', 'ATP binding', 'GTP binding',
                               'Amide binding', 'Antigen binding', 'Carbohydrate binding',
                               'Carbohydrate derivative binding', 'Chromatin binding', 'Hormone binding',
                               'Lipid binding', 'Nucleic acid binding', 'DNA binding', 'RNA binding',
                               'Organic cyclic compound binding', 'Peptide binding', 'Protein binding',
                               'Protein-containing complex binding', 'Small molecule binding',
                               'Sulfur compound binding', 'Catalytic activity', 'Cyclase activity',
                               'Demethylase activity', 'Hydrolase activity',
                               'Ribonucleoside triphosphate phosphatase activity', 'ATP hydrolysis activity',
                               'Gtpase activity', 'Isomerase activity', 'Ligase activity', 'Lyase activity',
                               'Oxidoreductase activity', 'Transferase activity',
                               'Catalytic activity, acting on a nucleic acid',
                               'Catalytic activity, acting on a protein', 'Electron transfer activity',
                               'Molecular adaptor activity', 'Protein-macromolecule adaptor activity',
                               'Molecular carrier activity', 'Nucleocytoplasmic carrier activity',
                               'Molecular transducer activity', 'Cytoskeletal motor activity', '',
                               'Molecular sequestering activity', 'Protein folding chaperone', 'Receptor activity',
                               'Cargo receptor activity', 'Signaling receptor activity', 'Regulator activity',
                               'Molecular function regulator activity', 'Atpase regulator activity',
                               'Enzyme regulator activity', 'Small molecule sensor activity',
                               'Signaling receptor regulator activity', 'Transporter regulator activity',
                               'Transcription regulator activity', 'DNA-binding transcription factor activity',
                               'Transcription coregulator activity', 'Translation regulator activity',
                               'Structural molecule activity', 'Transporter activity', 'Lipid transporter activity',
                               'Transmembrane transporter activity', 'Carrier activity', 'Channel activity']
        with open("../data/WordTableGO65.pkl", 'rb') as f:
            self.function_table = pickle.load(f)

        function2words = defaultdict(set)
        for word, functions in self.function_table.items():
            for function in functions:
                function2words[function].add(word)

    def predict(self, datalist):
        result = []
        for data in tqdm(datalist):
            functions = []
            match = {}
            for word in data[3]:
                word_function = self.function_table.get(word, {})
                if len(word_function) > 0:
                    match[word] = word_function
                    functions.extend(list(word_function))
            functions = self.filter_and_convert_to_set(functions, n=2)
            pred_label = [1 if i in functions else 0 for i in self.function_label]
            pred_label = self.adjust_pred_label(pred_label)
            pred_functions = [self.function_label[i] for i in range(len(pred_label)) if pred_label[i] == 1]

            result.append({
                'Uniprot ID': data[0],
                'Seq': data[1],
                'Pred function': pred_functions,
                'Match': match
            })
        return result

    @staticmethod
    def filter_and_convert_to_set(lst, n=2):
        count = Counter(lst)
        filtered_elements = [item for item in lst if count[item] >= n]
        result_set = set(filtered_elements)
        return result_set

    @staticmethod
    def adjust_pred_label(pred_label):
        rules = [
            {"range": (3, 22), "index_to_modify": 2},
            {"range": (4, 6), "index_to_modify": 3},
            {"range": (14, 16), "index_to_modify": 13},
            {"range": (23, 36), "index_to_modify": 22},
            {"range": (26, 29), "index_to_modify": 25},
            {"range": (27, 29), "index_to_modify": 26},
            {"range": (38, 39), "index_to_modify": 37},
            {"range": (40, 41), "index_to_modify": 39},
            {"range": (44, 46), "index_to_modify": 43},
            {"range": (47, 49), "index_to_modify": 46},
            {"range": (50, 60), "index_to_modify": 49},
            {"range": (51, 56), "index_to_modify": 50},
            {"range": (57, 59), "index_to_modify": 56},
            {"range": (62, 66), "index_to_modify": 61},
            {"range": (64, 66), "index_to_modify": 63},
        ]
        for rule in rules:
            start, end = rule['range']
            index_to_modify = rule["index_to_modify"]
            if any(pred_label[i] == 1 for i in range(start, end)):
                pred_label[index_to_modify] = 1
        return pred_label


if __name__ == '__main__':
    predictor = Predict()

    with open('../data/DUF_sample_1000.pkl', 'rb') as f:
        datalist = pickle.load(f)
    results = predictor.predict(datalist)
    for result in results:
        print(result)
