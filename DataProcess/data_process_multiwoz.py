import os
import json
import pandas as pd
from collections import OrderedDict
from collections import defaultdict
from fix_label import fix_general_label_error

class MultiWOZ:
    def __init__(self,data_dir,output_dir,onlogy_path):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.onlogy_path = onlogy_path
        self.init_everything()
    def init_everything(self):
        self.file_train = os.path.join(self.data_dir,'train_dials.json')
        self.file_dev = os.path.join(self.data_dir, 'dev_dials.json')
        self.file_test = os.path.join(self.data_dir, 'test_dials.json')
        self.ontology = json.load(open(self.onlogy_path, 'r'))
        self.DOMAIN_SLOT_VALUES = dict([(k, v) for k, v in self.ontology.items()])
        self.DOMAIN_SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in self.DOMAIN_SLOT_VALUES.keys()]
        self.SLOT_GATE = {"ptr": 0, "dontcare": 1, "none": 2}


    def read_examples(self,file_name, dataset,training=False):
        print(("Reading from {}".format(file_name)))
        data = []
        # 对话的id
        dialogue_idx_list = []
        # 每一轮的ID
        turn_idx_list = []
        # 当前轮的对话历史
        dialog_history_list = []
        turn_domain_list = []
        turn_uttrance_list=[]
        turn_history_list = []
        turn_belief_state_list = []

        with open(file_name) as f:
            dials = json.load(f)

            for dia_index,dial_dict in enumerate(dials):
                dialog_history = ""
                for turn_index, turn in enumerate(dial_dict["dialogue"]):
                    # 每一轮的表述
                    turn_uttrance = turn["system_transcript"].strip() + "#" + turn["transcript"].strip()
                    # 当前轮的对话历史
                    dialog_history += (turn["system_transcript"].strip() + "#" + turn["transcript"].strip()+ "#")
                    # 当前轮的对话状态，也就是我们要追踪的东西
                    turn_belief_dict = fix_general_label_error(turn["belief_state"], False, self.DOMAIN_SLOTS)
                    turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items()])
                    turn_belief_list = [str(k) + '-' + str(v) for k, v in turn_belief_dict.items()]

                    dialogue_idx_list.append(dial_dict["dialogue_idx"].strip())
                    turn_idx_list.append(turn["turn_idx"])
                    turn_domain_list.append(turn["domain"].strip())
                    turn_uttrance_list.append(turn_uttrance)
                    turn_history_list.append(dialog_history)
                    turn_belief_state_list.append(turn_belief_list)

        data_dict = {
            'DialogID':dialogue_idx_list,
            'TurnID':turn_idx_list,
            '本轮对话涉及的领域':turn_domain_list,
            '当前对话内容':turn_uttrance_list,
            '历史对话内容':turn_history_list,
            '当前对话状态':turn_belief_state_list,
        }
        data_example = pd.DataFrame(data_dict)
        return data_example



    def load_data(self):

        train_examples = self.read_examples(self.file_train, "train",training=True)
        dev_examples= self.read_examples(self.file_dev, "dev")
        test_examples  = self.read_examples(self.file_test, "test")

        print(train_examples.shape)
        train_examples.to_csv(os.path.join(self.output_dir, "train.csv"),index = False)
        dev_examples.to_csv(os.path.join(self.output_dir, "dev.csv"),index = False)
        test_examples.to_csv(os.path.join(self.output_dir, "test.csv"),index = False)
        df = pd.read_csv(os.path.join(self.output_dir, "train.csv"),nrows=50)
        df.to_csv(os.path.join(self.output_dir, "train_demo.csv"),index = False)



        print("Read %s examples train" % len(train_examples))
        print("Read %s examples dev" % len(dev_examples))
        print("Read %s examples test" % len(test_examples))



if __name__ == "__main__":
    trainer = MultiWOZ(
        data_dir='/home/lsy2018/graphDialog/data',
        output_dir = '/home/lsy2018/小论文/data/MultiWOZ',
        onlogy_path= '/home/lsy2018/graphDialog/data/multi-woz/MULTIWOZ2 2/ontology.json'
    )
    trainer.load_data()