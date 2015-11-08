## Entity Mention Detection using RNN and LSTM<br />
--------------------------------------------------------

The task details can be accessed at: www.biocreative.org/tasks/biocreative-v/cemp-detailed-task-description/

*To execute it,* <br />
**cd code/** <br />
**th code/main.lua** <br />
<br />
*To know configurable parameters of the model,* <br />
**th code/main.lua -help** <br />
<br />
*Prerequisites to run:* <br />
1. Torch 7 <br />
2. nn <br />
3. optim <br />
4. xlua <br />
5. rnn <br />
6. cunn (if you are running in a gpu) <br />
Packages [2-5] can be installed using Luarocks. <br />
For example,<br />
*To install nn,* <br />
luarocks install nn <br />
