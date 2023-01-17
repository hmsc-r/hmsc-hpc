import json

def load_model_from_json(json_file_path):

    with open(json_file_path) as json_file:
        hmsc_model = json.load(json_file)

    return hmsc_model

def save_postList_to_json(postList, path, nChains):

    json_data = {chain: {} for chain in range(nChains)}

    for chain in range(nChains):
        for i in range(len(postList[chain])):
            sample_data = {}

            sample_data["Beta"] = (
                postList[chain][i]["BetaLambda"]["Beta"].numpy().tolist()
            )
            sample_data["Gamma"] = postList[chain][i]["GammaV"]["iV"].numpy().tolist()

            sample_data["iV"] = postList[chain][i]["GammaV"]["iV"].numpy().tolist()
            sample_data["sigma"] = postList[chain][i]["sigma"].numpy().tolist()
            sample_data["Lambda"] = [
                postList[chain][i]["BetaLambda"]["Lambda"][j].numpy().tolist()
                for j in range(len(postList[chain][i]["BetaLambda"]["Lambda"]))
            ]
            sample_data["Eta"] = [
                postList[chain][i]["Eta"][j].numpy().tolist()
                for j in range(len(postList[chain][i]["Eta"]))
            ]
            sample_data["Psi"] = [
                postList[chain][i]["PsiDelta"]["Psi"][j].numpy().tolist()
                for j in range(len(postList[chain][i]["PsiDelta"]["Psi"]))
            ]
            sample_data["Delta"] = [
                postList[chain][i]["PsiDelta"]["Delta"][j].numpy().tolist()
                for j in range(len(postList[chain][i]["PsiDelta"]["Delta"]))
            ]

            sample_data["Alpha"] = [
                postList[chain][i]["Alpha"][j].numpy().tolist()
                for j in range(len(postList[chain][i]["Alpha"]))
            ]
            postList[chain][i]["BetaLambda"]["Beta"].numpy().tolist()

            sample_data["wRRR"] = sample_data["rho"] = sample_data[
                "PsiRRR"
            ] = sample_data["DeltaRRR"] = None

            json_data[chain][i] = sample_data

    with open(path + "obj-postList.json", "w") as fp:
        json.dump(json_data, fp)