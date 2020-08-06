from keras.models import load_model
import sys

model_path=sys.argv[1]
test_path=sys.argv[2]

model=load_model(model_path)

