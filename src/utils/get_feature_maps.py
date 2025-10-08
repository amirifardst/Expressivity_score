
import os
import tarfile
from src.logging.logger import get_logger
from nats_bench import create
from nats_bench.api_utils import time_string
from xautodl.models import get_cell_based_tiny_net,get_cifar_models
import torchinfo
import requests
from tqdm import tqdm
from tensorflow.keras.models import Model
make_logger = get_logger(__name__)

# Download dataset of NATS-tss-v1_0-3ffb9-simple.tar
def connect_api():
    """
    This function downloads and connects to the NATS-Bench API.
    Returns:
        API: The NATS-Bench API instance.
    """
    # Download dataset
    if not os.path.exists('dataset/NATS-tss-v1_0-3ffb9-simple'):
        make_logger.info("Downloading dataset of NATS-bench...")
        url = "https://drive.usercontent.google.com/download?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU&export=download&authuser=0&confirm=t&uuid=b9be6591-4e48-418a-8418-58949b31a70b&at=AN8xHoozl3AS3QVskRyGg7-s4mEI:1756387264341"
        output_file = "NATS-tss-v1_0-3ffb9-simple.tar"
        response = requests.get(url, stream=True, verify=False)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(output_file, "wb") as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
            make_logger.info("Download complete!")

            # Extract the .tar file into the dataset folder
            extraction_path = os.path.join("dataset")
            os.makedirs(extraction_path, exist_ok=True)
            with tarfile.open(output_file) as tar:
                tar.extractall(path=extraction_path)
            make_logger.info(f"Extraction complete! Files are extracted to {extraction_path}")
        else:
            make_logger.error("Failed to download file:", response.status_code)

    else:
        make_logger.info("Dataset already exists. Skipping download.")
    # Create the API for size search space
    os.environ['TORCH_HOME'] = "./dataset"
    API = create(None, 'tss', fast_mode=True, verbose=False)
    make_logger.info(f"API created with {len(API)} architectures successfully.")
    return API

def show_model_summary(API, model_number, database='cifar10'):
    """
    This function shows the summary of a model from the NATS-Bench API.
    Args:
        model_number (int): The model number to retrieve.
        database (str): The database to retrieve the model from.
    """
    # Instantiates a model from the NATS-Bench API.
    config = API.get_net_config(model_number, database)
    model = get_cell_based_tiny_net(config)

    # Show model summary
    # summary(model, input_size=(3, 32, 32))
    torchinfo.summary(model,depth=20, input_size=(1, 3, 32, 32), col_names=["input_size", "output_size", "num_params", "trainable"], row_settings=["var_names"])

def get_layer_fmap_list(feature_maps, model):
    """
    This function retrieves the feature maps for each layer in the model.
    Args:
        feature_maps (dict): A dictionary containing the feature maps for each layer.
        model (torch.nn.Module): The model from which to retrieve the feature maps.
    Returns:
        tuple: A tuple containing the sorted layer names, layer details, and feature maps.
    """
    # Get modules in traversal order
    modules_ordered = list(model.modules())
    modules_with_maps = [m for m in modules_ordered if m in feature_maps]

    layer_names_sorted = []
    layer_details = []
    fmap_list = []
  
    for i, module in enumerate(modules_with_maps):
        fmap = feature_maps[module]
        layer_name = f"Layer_{i}_{module.__class__.__name__}"
        layer_detail = f"layer_{i}_{module}"
        fmap_np = fmap.detach().cpu().numpy().astype('float16')

        layer_names_sorted.append(layer_name)
        layer_details.append(layer_detail)
        fmap_list.append(fmap_np)



    return (layer_names_sorted, layer_details, fmap_list)

def extract_feature_maps_evaluation(API, num_models_to_evaluate, database, random_input):
    """
    This function extracts feature maps for multiple models from a database in evaluation mode
    Args:
        API
        num_models_to_evaluate (int): The number of models to evaluate.
        database (str): The database to extract models from.
        random_input (torch.Tensor): A random input tensor to pass through the model.
    Returns:
        model_names_list (list): A list of model names.
        ground_truth_acc_list (list): A list of ground truth accuracies.
        fmap_dict (dict): A dictionary containing the feature maps for each model.
    """
    make_logger.info(f"Extracting feature maps for {num_models_to_evaluate} models from database {database} starting...")
    ground_truth_acc_list = []
    model_names_list = []
    fmap_dict = {}
    for model_id in range(num_models_to_evaluate):
        print(f"Extracting feature maps for model {model_id} from database {database}")
        # Instantiates a model from the NATS-Bench API.
        config = API.get_net_config(model_id, database)
        model = get_cell_based_tiny_net(config)

        feature_maps = {}

        def hook_fn(module, input, output):
            feature_maps[module] = output
        hooks = []
        for module in model.modules():
            if len(list(module.children())) == 0:
                h = module.register_forward_hook(hook_fn)
                hooks.append(h)

        _ = model(random_input)

        for h in hooks:
            h.remove()


        test_accuracy = round(float(API.get_more_info(model_id, database, hp='200', is_random=False)['test-accuracy']), 2)
        ground_truth_acc_list.append(test_accuracy)

        model_key = f"model_{model_id:03d}"
        model_names_list.append(model_key)

        layer_names_sorted, layer_details, fmap_list = get_layer_fmap_list(feature_maps, model)

        fmap_dict[model_key] = {
            'layer_names': layer_names_sorted,
            'layer_details': layer_details,
            'fmap_list': fmap_list
        }

    make_logger.info(f"Feature maps extraction for {num_models_to_evaluate} models from database {database} completed.")
    return model_names_list, ground_truth_acc_list, fmap_dict

# Function to extract feature maps from the model
def extract_feature_maps_from_torch(model, x):
    """
    Extract feature maps from all leaf layers in the model.
    
    Args:
        model: Keras model instance.
        x: Input numpy array or tensor with shape (batch_size, h, w, c).
        
    Returns:
        Dictionary mapping layer name to output feature map numpy array.
    """
    fmap_dict= {}
    fmap_list = []
    
    def hook(module, input, output):
        fmap_list.append(output)
    
    # Register hooks to capture the output of each layer
    hooks = []
    layer_names_sorted = []
    layer_details = []
    for i, layer in enumerate(model.children()):
        layer_names_sorted.append(f"Layer_{i}")
        layer_detail = f"Layer_{i}_{layer.__class__.__name__}"
        layer_details.append(layer_detail)
        hooks.append(layer.register_forward_hook(hook))
    
    # Forward pass through the model
    model(x)
    
    # Remove hooks after capturing the feature maps
    for h in hooks:
        h.remove()
    

    fmap_dict['input_model'] = {
            'layer_names': layer_names_sorted,
            'layer_details': layer_details,
            'fmap_list': fmap_list}
    return fmap_dict

def extract_feature_maps_from_keras(model, x):
    """
    Extract feature maps from all leaf layers in the model.
    
    Args:
        model: Keras model instance.
        x: Input numpy array or tensor with shape (batch_size, h, w, c).
        
    Returns:
        Dictionary mapping layer name to output feature map numpy array.
    """
    # Initialize dictionaries and lists to store feature maps and layer details
    fmap_dict = {}
    fmap_list = []
    layer_names_sorted = []
    layer_details = []

    # Iterate through all layers in the model
    for i, layer in enumerate(model.layers):
        layer_name = f"Layer_{i}"
        layer_detail = f"Layer_{i}_{layer.__class__.__name__}"
        layer_names_sorted.append(layer_name)
        layer_details.append(layer_detail)

        # Create a sub-model for the current layer to extract its output
        intermediate_model = Model(inputs=model.input, outputs=layer.output)

        # Run forward pass to get the feature map for the current layer
        feature_map = intermediate_model.predict(x)
        # shape = feature_map.shape
        fmap_list.append(feature_map)

    # Store the extracted feature maps and layer details in the dictionary
    fmap_dict['input_model'] = {
        'layer_names': layer_names_sorted,
        'layer_details': layer_details,
        'fmap_list': fmap_list
    }

    return fmap_dict
