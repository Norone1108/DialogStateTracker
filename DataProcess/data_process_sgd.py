import os
import json
import pandas as pd
from collections import OrderedDict
from collections import defaultdict


class GoolgeSGD:
    def __init__(self,data_dir,output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir




    def read_examples(self,file_name, dataset):
        file_path = os.path.join(file_name,dataset)
        # print(os.listdir(file_path))
        print(("Reading from {}".format(file_path)))

        json_file_list = []
        dialog_idx_list = []
        turn_idx_list = []
        frames_list = []
        history_list = []
        speaker_list = []
        utterance_list = []
        turn_belief_state_list = []
        for each_json in os.listdir(file_path):
            dialog = json.load(open(os.path.join(file_path,each_json), 'r'))
            # print(dialog)
            for each_dialog in dialog:
                history = ''
                if 'turns' not in each_dialog:continue # 先不处理schema
                for turn_index,each_turn in enumerate(each_dialog['turns']):
                    print(turn_index)

                    history+=each_turn['utterance']+'#'
                    if each_turn['speaker'] == 'SYSTEM':continue
                    belief_state = []
                    for each_frame in each_turn['frames']:
                        if 'service' in each_frame:
                            domain  = each_frame['service']
                            slot_values = each_frame['state']['slot_values']
                            for each_slot in slot_values:
                                for each_value in slot_values[each_slot]:
                                    each_state = '#'.join([domain,each_slot,each_value])
                                    belief_state.append(each_state)

                    json_file_list.append(each_json)
                    dialog_idx_list.append(each_dialog['dialogue_id'])
                    turn_idx_list.append(turn_index)
                    frames_list.append(each_turn['frames'])
                    speaker_list.append(each_turn['speaker'])
                    utterance_list.append((each_turn['utterance']))
                    history_list.append(history)
                    turn_belief_state_list.append(belief_state)

        data_dict = {
            'Json文件':json_file_list,
            'DialogID':dialog_idx_list,
            'TurnID':turn_idx_list,
            '本轮对话涉及的领域':frames_list,
            '当前说话人':speaker_list,
            '当前对话内容':utterance_list,
            '历史对话内容':history_list,
            '当前对话状态':turn_belief_state_list,
        }
        data_example = pd.DataFrame(data_dict)
        return data_example



    def load_data(self):

        train_examples = self.read_examples(self.data_dir, "train")
        dev_examples= self.read_examples(self.data_dir, "dev")
        test_examples  = self.read_examples(self.data_dir, "test")

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
    trainer = GoolgeSGD(
        data_dir='/home/lsy2018/小论文/data/GoolgeSGD/dstc8-schema-guided-dialogue-master',
        output_dir = '/home/lsy2018/小论文/data/GoolgeSGD'
    )
    trainer.load_data()