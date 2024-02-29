from preprocessing import Preprocessor
from training import Trainer

if __name__=="__main__":
    current_time = Preprocessor()()
    Trainer(current_time)()