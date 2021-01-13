import os

def setup_endpoints(fpath:str):
    os.chdir(fpath)
    if not os.path.isdir("results"):
        os.mkdir("results")
    if not os.path.isdir("logs"):
        os.mkdir("logs")
    if not os.path.isdir("data"):
         os.mkdir("data")

def write_feature_and_targets(X, Y, fpath:str):
    """
    Takes the X and Y and writest them to a file
    """
    with open(fpath+'/data/all_feature.txt', 'w') as file_name:
        for feature in X:
            file_name.write(str(feature) + ',' + '\n')
        for target in Y:
            file_name.write(str(target) + ',' + '\n')
