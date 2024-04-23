from NN_tutorial import *
class SkipConnectionModel(nn.Module):
    def __init__(self, input_dim, output_dim, depth, width):
        super(SkipConnectionModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, width) 
        self.relu = nn.ReLU()
        self.linear_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth-1)])
        self.relu_layers = nn.ModuleList([nn.ReLU() for _ in range((int)(depth/2))])
        self.output_linear = nn.Linear(width, output_dim)
        self.depth = depth

    def forward(self, x):

        first_output = self.input_linear(x) 
        output = self.relu(first_output)
        skip_output = output + first_output
        # the blocks of layers with skip connections
        for i in range(len(self.relu_layers)):
            output = self.linear_layers[2*i](output)
            output = self.relu_layers[i](output) 
            if 2*i+1<self.depth-1:
                output= self.linear_layers[2*i+1](output)
            output = output+skip_output
            skip_output = output
        
        return self.output_linear(output)