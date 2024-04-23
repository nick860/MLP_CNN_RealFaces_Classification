from NN_tutorial import *
from xg import *
from cnn import *
from skip import SkipConnectionModel


def create_model(input_dim,output_dim, depth, width, batch_norm):
    model = []
    if batch_norm:
        model.append(nn.Linear(input_dim, width))
        model.append(nn.BatchNorm1d(width))
        model.append(nn.ReLU())
    else:
        model.append(nn.Linear(input_dim, width))
        model.append(nn.ReLU())
    
    for i in range(depth-1):
        if batch_norm:
            model.append(nn.Linear(width, width))
            model.append(nn.BatchNorm1d(width))
            model.append(nn.ReLU())
        else:
            model.append(nn.Linear(width, width))
            model.append(nn.ReLU())
    model.append(nn.Linear(width, output_dim))
    model = nn.Sequential(*model)
    return model

def implict_representation(x_train, x_val, x_test, output_dim):
    model = create_model(20, output_dim, 6, 16, False)
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, magnitude = train_model(x_train, x_val, x_test, model, lr=0.001, epochs=50, batch_size=256, implict = True)
    plot_decision_boundaries(model, x_test[['long', 'lat']].values, x_test['country'].values, title='Decision Boundaries with implict', implicit_repr=True)
    print("The accuracy of the model is: ", test_accs[-1])
    model2 = create_model(2, output_dim, 6, 16, False)
    model2, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, magnitude = train_model(x_train, x_val, x_test, model2, lr=0.001, epochs=50, batch_size=256)
    plot_decision_boundaries(model2, x_test[['long', 'lat']].values, x_test['country'].values, title='Decision Boundaries without implict', implicit_repr=False)
    print("The accuracy of the model is: ", val_accs[-1])

def train_the_modles(train_data, val_data, test_data, output_dim, input_dim,):
    depth = [1,2,6,10,6,6,6] # the number of hidden layers
    width = [16,16,16,16,8,32,64] # the number of neurons in the hidden layers
    # list of learning rates
    lr_list = [0.001, 0.001, 0.001, 0.0005, 0.001, 0.001, 0.0005]
    # list of ephocs 
    epochs_list = [50, 60, 100, 110, 90, 100, 105]

    save_model = []
    train_accuracy_list = []
    val_accuracy_list = []
    test_accuracy_list = []
    the_losses = []
    for i in range(7):
        model = create_model(input_dim, output_dim, depth=depth[i], width=width[i], batch_norm=True)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, magnitude= train_model(train_data, val_data, test_data, model, lr=lr_list[i], epochs=epochs_list[i], batch_size=256)
        save_model.append(model)
        train_accuracy_list.append(train_accs[-1])
        val_accuracy_list.append(val_accs[-1])
        test_accuracy_list.append(test_accs[-1])
        the_losses.append([train_losses, val_losses, test_losses])

    # the best model
    best_model_index = np.argmax(val_accuracy_list)
    best_model = save_model[best_model_index]
    plt.plot(range(epochs_list[best_model_index]), the_losses[best_model_index][0], label="train loss")
    plt.plot(range(epochs_list[best_model_index]), the_losses[best_model_index][1], label="validation loss")
    plt.plot(range(epochs_list[best_model_index]), the_losses[best_model_index][2], label="test loss")
    plt.legend()
    plt.show()
     
    plot_decision_boundaries(best_model, test_data[['long', 'lat']].values, test_data['country'].values, title='Best Decision Boundaries', implicit_repr=False)

    # the worst model
    worst_model_index = np.argmin(val_accuracy_list)
    worst_model = save_model[worst_model_index]
    plt.plot(range(epochs_list[worst_model_index]), the_losses[worst_model_index][0], label="train loss")
    plt.plot(range(epochs_list[worst_model_index]), the_losses[worst_model_index][1], label="validation loss")
    plt.plot(range(epochs_list[worst_model_index]), the_losses[worst_model_index][2], label="test loss")
    plt.legend()
    plt.show()
    
    plot_decision_boundaries(worst_model, test_data[['long', 'lat']].values, test_data['country'].values, title='Worst Decision Boundaries', implicit_repr=False)

    """Using only the MLPs of width 16, plot the training,
validation and test accuracy of the models vs. number of hidden layers. (x
axis - number of hidden layers, y axis - accuracy)"""
    plt.plot([1,2,6,10], train_accuracy_list[:4], label="Train accuracy")
    plt.plot([1,2,6,10], val_accuracy_list[:4], label="Validation accuracy")
    plt.plot([1,2,6,10], test_accuracy_list[:4], label="Test accuracy")
    plt.legend()
    plt.show()

    plt.plot([8,16,32,64], [train_accuracy_list[i] for i in [4,2,5,6]], label="Train accuracy")
    plt.plot([8,16,32,64], [val_accuracy_list[i] for i in [4,2,5,6]], label="Validation accuracy")
    plt.plot([8,16,32,64], [test_accuracy_list[i] for i in [4,2,5,6]], label="Test accuracy")
    plt.legend()
    plt.show()


def monitor_gradients(train_data, val_data, test_data, output_dim, input_dim, bonus= False):
    depth = 100 # the number of hidden layers
    width = 4
    epochs_num = 10
    if bonus: 
        model = SkipConnectionModel(input_dim, output_dim, depth, width)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, magnitude = train_model(train_data, val_data, test_data, model, lr=0.001, epochs=epochs_num, batch_size=256, do_clip_grad_norm_=True)
    else:
        model = create_model(input_dim, output_dim, depth=100, width=4, batch_norm=False)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, magnitude = train_model(train_data, val_data, test_data, model, lr=0.001, epochs=epochs_num, batch_size=256)

    plt.plot(range(epochs_num), magnitude[0], label="Layer 0")
    plt.plot(range(epochs_num), magnitude[30], label="Layer 30")
    plt.plot(range(epochs_num), magnitude[60], label="Layer 60")
    plt.plot(range(epochs_num), magnitude[90], label="Layer 90")
    plt.plot(range(epochs_num), magnitude[95], label="Layer 95")
    plt.plot(range(epochs_num), magnitude[99], label="Layer 99")
    plt.legend()
    plt.show()


def different_learning_rates(input_dim,output_dim, train_data, val_data, test_data):
    val_losses_list = []
    learning_rates = [1, 0.01, 0.001, 0.00001]
    for i in learning_rates:
        model = create_model(input_dim, output_dim, depth=6, width=16, batch_norm=False)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, magnitude= train_model(train_data, val_data, test_data, model, lr=i, epochs=50, batch_size=256)
        val_losses_list.append(val_losses)
    for i in range(4):
        plt.plot(range(50), val_losses_list[i], label=learning_rates[i])
    
    plt.title('Validation Losses')
    plt.legend()
    plt.show()

def different_ephocs(input_dim,output_dim, train_data, val_data, test_data):
    model = create_model(input_dim, output_dim, depth=6, width=16, batch_norm=False)
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, magnitude = train_model(train_data, val_data, test_data, model, lr=0.001, epochs=100, batch_size=256)
    epochs = [1,5,10,20,50,100]
    ephocs_loss = [val_losses[i-1] for i in epochs]
    plt.scatter(epochs, ephocs_loss, label="losses")
    plt.title('Validation Losses')
    plt.legend()
    plt.show()

def batch_norm(input_dim,output_dim, train_data, val_data, test_data):
    model = create_model(input_dim, output_dim, depth=6, width=16, batch_norm=True)
    model, train_accs, val_accs, test_accs_norm, train_losses, val_losses, test_losses, magnitude= train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256)
    plt.plot(range(50), val_losses, label="Batch norm")
    model = create_model(input_dim, output_dim, depth=6, width=16, batch_norm=False)
    model, train_accs, val_accs, test_accs_without_norm, train_losses, val_losses, test_losses, magnitude= train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256)
    plt.plot(range(50), val_losses, label="Without batch norm")
    plt.title('Validation Losses')
    plt.legend()
    plt.show()
    # plot the test accuracies
    plt.plot(range(50), test_accs_norm, label="Batch norm")
    plt.plot(range(50), test_accs_without_norm, label="Without batch norm")
    plt.title('Test Accuracies')
    plt.legend()
    plt.show()

def different_batch_sizes(input_dim,output_dim, train_data, val_data, test_data):
    batch_sizes = [1, 16, 128, 1024]
    epochs = [1,10,50,50]
    val_losses_list = []
    test_accs_list = []
    for i in range(4):
        model = create_model(input_dim, output_dim, depth=6, width=16, batch_norm=False)
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses, magnitude= train_model(train_data, val_data, test_data, model, lr=0.001, epochs=epochs[i], batch_size=batch_sizes[i])
        val_losses_list.append(val_losses)
        test_accs_list.append(test_accs)

    plt.scatter(range(epochs[0]), val_losses_list[0], label="batch size " + str(batch_sizes[0]))
    plt.plot(range(epochs[1]), val_losses_list[1], label="batch size " + str(batch_sizes[1]))
    plt.plot(range(epochs[2]), val_losses_list[2], label="batch size " + str(batch_sizes[2]))
    plt.plot(range(epochs[3]), val_losses_list[3], label="batch size " + str(batch_sizes[3]))

    plt.title('Validation Losses')
    plt.legend()
    plt.show()

    plt.scatter(range(epochs[0]), test_accs_list[0], label="batch size " + str(batch_sizes[0]))
    plt.plot(range(epochs[1]), test_accs_list[1], label="batch size " + str(batch_sizes[1]))
    plt.plot(range(epochs[2]), test_accs_list[2], label="batch size " + str(batch_sizes[2]))
    plt.plot(range(epochs[3]), test_accs_list[3], label="batch size " + str(batch_sizes[3]))

    plt.title('Test Accuracies')
    plt.legend()
    plt.show()

def MLP():
    torch.manual_seed(0)
    np.random.seed(0)

    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')


    output_dim = len(train_data['country'].unique())
    input_dim = 2
    
    print("doing the first part")
    different_learning_rates(input_dim, output_dim, train_data, val_data, test_data)
    print("doing the second part")
    different_ephocs(input_dim, output_dim, train_data, val_data, test_data)
    print("doing the third part")
    batch_norm(input_dim, output_dim, train_data, val_data, test_data)
    print("doing the fourth part")
    different_batch_sizes(input_dim, output_dim, train_data, val_data, test_data)
    print("doing the fifth part")
    train_the_modles(train_data, val_data, test_data, output_dim, input_dim)
    print("doing the sixth part")
    monitor_gradients(train_data, val_data, test_data, output_dim, input_dim)
    print("doing the seventh part")
    monitor_gradients(train_data, val_data, test_data, output_dim, input_dim, bonus=True) # bonus
    print("doing the eight part")
    implict_representation(train_data, val_data, test_data, output_dim) # bonus

def bonus():
        torch.manual_seed(0)
        #fine tuning
        model_scratch = ResNet18(pretrained=True, probing=False)
        train_baseline(model_scratch, 200, 0.0001, sklearn=False, bonus=True)
        
def Cnn_and_xg_baselines():
    torch.manual_seed(0)
    np.random.seed(0)
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    baselines_test_accuracy = {"XGBoost": [], "Scratch": [], "Linear probing": [], "Linear sklearn": [], "Fine-tuning": []}
    baselines_models = {"XGBoost": [], "Scratch": [], "Linear probing": [], "Linear sklearn": [], "Fine-tuning": []}
    baselines_predicted = {"XGBoost": [], "Scratch": [], "Linear probing": [], "Linear sklearn": [], "Fine-tuning": []}

    for lr in learning_rates:
        #1 XGBoost
        xgb_accuracy , model_xgb , predicted_xgb = train_xgboost(lr)
        baselines_test_accuracy["XGBoost"].append(xgb_accuracy)
        baselines_models["XGBoost"].append(model_xgb)
        baselines_predicted["XGBoost"].append(predicted_xgb)
        #2 Scratch
        model_scratch = ResNet18(pretrained=False, probing=False)
        acc, predicted_scrach = train_baseline(model_scratch, 1, lr)
        baselines_test_accuracy["Scratch"].append(acc)
        baselines_models["Scratch"].append(model_scratch)
        baselines_predicted["Scratch"].append(predicted_scrach)
        # show the image
        #3 Linear probing
        model_linear_probing = ResNet18(pretrained=True, probing=True)
        acc, predicted_linear_probing = train_baseline(model_linear_probing, 1, lr)
        baselines_test_accuracy["Linear probing"].append(acc)
        baselines_models["Linear probing"].append(model_linear_probing)
        baselines_predicted["Linear probing"].append(predicted_linear_probing)
        #4 Linear sklearn - bonus
        model_sklearn= ResNet18(pretrained=True, probing=True) # doesn't matter about the probing
        acc, predicted_linear_sklearn = train_baseline(model_sklearn, 1, lr, sklearn=True)
        baselines_test_accuracy["Linear sklearn"].append(acc)
        baselines_models["Linear sklearn"].append(model_sklearn)
        baselines_predicted["Linear sklearn"].append(predicted_linear_sklearn)
        #5 Fine-tuning
        model_fine_tuning = ResNet18(pretrained=True, probing=False)
        acc, predicted_fine_tuning = train_baseline(model_fine_tuning, 1, lr)
        baselines_test_accuracy["Fine-tuning"].append(acc)
        baselines_models["Fine-tuning"].append(model_fine_tuning)
        baselines_predicted["Fine-tuning"].append(predicted_fine_tuning)
    
    the_baselines = []
    for key in baselines_test_accuracy:

        # the best index
        the_best = np.argmax(baselines_test_accuracy[key])
        # the second best index
        the_second_best = np.argsort(baselines_test_accuracy[key])[-2]
        # the worst index
        the_worst = np.argmin(baselines_test_accuracy[key])
        the_baselines.append((key, the_best, the_second_best, the_worst))

    the_best_model = None
    max_accuracy = 0
    the_worst_modles = []
    the_worst_model = None
    min_accuracy = 1
    best_prediction = None
    worst_prediction = None
    for baseline in the_baselines:
        print(baseline[0])
        best_test_accuracy = baselines_test_accuracy[baseline[0]][baseline[1]] 
        second_best_test_accuracy = baselines_test_accuracy[baseline[0]][baseline[2]]
        worst_test_accuracy = baselines_test_accuracy[baseline[0]][baseline[3]]

        if best_test_accuracy > max_accuracy:
            max_accuracy = best_test_accuracy
            the_best_model = baselines_models[baseline[0]][baseline[1]]
            best_prediction = baselines_predicted[baseline[0]][baseline[1]]

        if worst_test_accuracy < min_accuracy:
            min_accuracy = worst_test_accuracy
            the_worst_model = baselines_models[baseline[0]][baseline[3]]
            worst_prediction = baselines_predicted[baseline[0]][baseline[3]]
                                                                
        the_worst_modles.append(baselines_models[baseline[0]][baseline[3]])
        print("The best learning rate: ", learning_rates[baseline[1]], " with accuracy: ", best_test_accuracy )
        print("The second best learning rate: ", learning_rates[baseline[2]], " with accuracy: ", second_best_test_accuracy)
        print("The worst learning rate: ", learning_rates[baseline[3]], " with accuracy: ", worst_test_accuracy)

    test_loader = get_loaders(os.path.join(os.getcwd(), 'whichfaceisreal'), transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), 32)[2]
    the_correct_lables = [test_loader.dataset[i][1] for i in range(len(test_loader.dataset))]
    indices = np.where((best_prediction == the_correct_lables) & (worst_prediction != the_correct_lables))[0][:5]
    for index in indices:
        plt.imshow(test_loader.dataset[index][0].permute(1, 2, 0))
        plt.show()

if __name__ == "__main__":
    MLP()
    Cnn_and_xg_baselines()
    bonus()