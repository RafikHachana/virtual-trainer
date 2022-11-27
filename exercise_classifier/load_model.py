import torch
from torch import nn
from torch.autograd import Variable 

NUM_CLASSES = 6

class VideoExerciseClassifier(nn.Module):
  def __init__(self):
    super().__init__()

    self.num_layers = 2

    self.hidden_size = 8
    self.input_size = 50

    self.relu = nn.ReLU()

    self.lstm = nn.LSTM(8, 8, num_layers=2, batch_first=True)
    self.dense = nn.Linear(self.hidden_size, NUM_CLASSES)
    # self.dense_2 = nn.Linear(12, NUM_CLASSES)


  def forward(self,x):
    h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
    c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
    # Propagate input through LSTM
    output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
    # print(hn.size())
    # hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
    out = self.relu(hn[-1])
    out = self.dense(out) #first Dense
    # out = self.relu(out) #relu
    # out = self.fc(out) #Final Output
    # print(out.size())
    return out
  

model = VideoExerciseClassifier()
model.load_state_dict(torch.load("./exercise_classifier.pt", map_location=torch.device('cpu')))
model.eval()


def predict_exercise(all_timeseries):
    sequence = []
    try:
      for i in range(50):
          vec = []

          for k, v, in all_timeseries.items():
              vec.append(v[i][1])

          sequence.append(vec)

    except IndexError:
       return "Unknown exercise (not enough frames to predict)"
       
    model_input = torch.unsqueeze(torch.tensor(sequence), 0)

    output = model(model_input)

    probs = nn.functional.softmax(output, dim=1)

    pred = torch.argmax(probs, dim=1)

    # print(pred)

    idx_to_labels = {0: 'Deadlift', 1: 'Biceps curl', 2: 'Push-up', 3: 'Tricep Pushdown', 4: 'Lat pulldown', 5: 'Incline bench press'}

    return idx_to_labels[pred[0].item()]


