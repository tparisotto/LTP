import sys
import re 
from util import get_list_from_textfile, eval_accuracy

input_path = "../data/audio_all.txt"
output_path = "../data/baseline.txt"
true_path = "../data/preprocessed_all.txt"

def baseline(input_path, output_path):
        f = open(input_path, "r", encoding="utf8")
        lines = f.readlines()
        word_list = []

        for i in range(0,len(lines)):
                words = re.findall(r"[\w'-]+|[.,!?;]", lines[i])
                for w in words:
                        #print(w)
                        word_list.append(w)

        for i in range(0,len(word_list)-1):
                if(     
                word_list[i] == "!" or 
                word_list[i] == "?" or 
                word_list[i] == "," or 
                word_list[i] == "." or 
                word_list[i] == ";"
                ):
                        word_list.insert(i+1,"<BRK>")

        f.close()
        baseline_file = open(output_path, "w", encoding="utf8")
        for w in word_list:
                baseline_file.write(w + " ")
        baseline_file.close()



baseline(input_path,output_path)
eval_accuracy(output_path,true_path)