import ujson as json
import os


def load_model_from_json(json_file_path):

    with open(json_file_path) as json_file:
        hmsc_obj = json.load(json_file)

    return hmsc_obj, hmsc_obj.get("hM")


def save_postList_to_json(postList, postList_file_path, chain):

    print("Start dumping.")

    json_data = {}

    for i in range(len(postList)):
        sample_data = {}
        par = postList[i]
        
        sample_data["Beta"] = par["Beta"].numpy().tolist()
        sample_data["Gamma"] = par["Gamma"].numpy().tolist()
        sample_data["V"] = par["V"].numpy().tolist()
        sample_data["sigma"] = par["sigma"].numpy().tolist()
        
        sample_data["Lambda"] = [
            postList[i]["Lambda"][j].numpy().tolist()
            for j in range(len(postList[i]["Lambda"]))
        ]
        sample_data["Eta"] = [
            postList[i]["Eta"][j].numpy().tolist()
            for j in range(len(postList[i]["Eta"]))
        ]
        sample_data["Psi"] = [
            postList[i]["Psi"][j].numpy().tolist()
            for j in range(len(postList[i]["Psi"]))
        ]
        sample_data["Delta"] = [
            postList[i]["Delta"][j].numpy().tolist()
            for j in range(len(postList[i]["Delta"]))
        ]

        sample_data["Alpha"] = [
            postList[i]["Alpha"][j].numpy().tolist()
            for j in range(len(postList[i]["Alpha"]))
        ]

        #sample_data["Lambda"] = sample_data["Eta"] = sample_data["Psi"] = sample_data["Delta"] = sample_data["Alpha"] = None
        sample_data["wRRR"] = sample_data["rho"] = sample_data["PsiRRR"] = sample_data["DeltaRRR"] = None

        json_data[i] = sample_data

    postList_file_path = (
        os.path.splitext(postList_file_path)[0] + "_" + str(chain+1) + ".json"
    )
    print("Dumping, chain %d" % chain)
    with open(postList_file_path, "w") as fp:
        json.dump(json_data, fp)


def save_chains_postList_to_json(postList, postList_file_path, nChains):

    json_data = {chain: {} for chain in range(nChains)}

    for chain in range(nChains):
        for i in range(len(postList[chain])):
            sample_data = {}
            par = postList[chain][i]

            sample_data["Beta"] = par["Beta"].numpy().tolist()
            sample_data["Gamma"] = par["Gamma"].numpy().tolist()
            sample_data["V"] = par["V"].numpy().tolist()
            sample_data["sigma"] = par["sigma"].numpy().tolist()
            
            
            sample_data["Lambda"] = [
                par["Lambda"][j].numpy().tolist()
                for j in range(len(par["Lambda"]))
            ]
            sample_data["Eta"] = [
                par["Eta"][j].numpy().tolist()
                for j in range(len(par["Eta"]))
            ]
            sample_data["Psi"] = [
                par["Psi"][j].numpy().tolist()
                for j in range(len(par["Psi"]))
            ]
            sample_data["Delta"] = [
                par["Delta"][j].numpy().tolist()
                for j in range(len(par["Delta"]))
            ]
            
            sample_data["Alpha"] = [
                par["Alpha"][j].numpy().tolist()
                for j in range(len(par["Alpha"]))
            ]
            
            # sample_data["Lambda"] = sample_data["Eta"] = sample_data["Psi"] = sample_data["Delta"] = sample_data["Alpha"] = None
            sample_data["wRRR"] = sample_data["rho"] = sample_data["PsiRRR"] = sample_data["DeltaRRR"] = None

            json_data[chain][i] = sample_data

    with open(postList_file_path, "w") as fp:
        json.dump(json_data, fp)
