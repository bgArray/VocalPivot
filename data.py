import os

import data_process
from Constant import *


if __name__ == '__main__':
    spath = os.path.join(DATA_PATH, L_Singer[1], "Mixed_Voice_and_Falsetto")
    techs = ["Control_Group",
        "Falsetto_Group",
        "Mixed_Voice_Group",]
    for i in os.listdir(spath):
        for j in techs:
            base_path = os.path.join(spath, i, j)
            data_process.GTS_LSTM_operator(base_path)
            # raise ValueError


    # data_process.GTS_LSTM_operator(r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Chinese\ZH-Alto-1\Mixed_Voice_and_Falsetto\修炼爱情\Control_Group", )
    # data_process.GTS_LSTM_operator(r"E:\文档_学习_高中\活动杂项\AiVocal\dataset\Chinese\ZH-Alto-1\Mixed_Voice_and_Falsetto\修炼爱情\Falsetto_Group")
