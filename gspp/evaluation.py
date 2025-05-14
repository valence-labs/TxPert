import torch
import numpy as np

import gspp.constants as cs


def evaluate(loader, model, device, adata, id2pert):
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    dose_cat = []
    pred = []
    truth = []
    cell_types = []
    experimental_batches = []
    results = {}

    pert_name_cache = {}
    # head obs is, e.g.
    # condition        control batch cell_type dose_val      condition_name
    # UBL5+ctrl        0       25    K562      1+1            K562_UBL5+ctrl_1+1
    # so the following makes a mapping from e.g. 'K562_UBL5+ctrl_1+1' to 'UBL5+ctrl' style naming
    for condition_name in np.unique(adata.obs[["condition_name"]]):
        pert_name_cache[condition_name] = (
            condition_name.split("_")[1],
            condition_name.split("_")[-1],
        )

    for batch in loader:
        with torch.no_grad():
            if model.no_pert:
                p = batch.control[:, : model.adata_output_dim]
            else:
                p = model.sample_inference(
                    batch.control, batch.pert_idxs, batch.p, batch.cell_types
                )
            t = batch.x[:, : model.adata_output_dim]
            pred.append(p.cpu())
            truth.append(t.cpu())
            for itr, perts in enumerate(batch.pert_cond_names):
                pert_name = pert_name_cache[perts][0]
                dose = pert_name_cache[perts][1]
                pert_cat.append(pert_name)
                dose_cat.append(dose)
                cell_types.append(batch.cell_types[itr])
                experimental_batches.append(batch.experimental_batches[itr])

    print(f"A total of {len(pert_cat)} perturbations were evaluated", flush=True)
    results[cs.RES_PERT_CAT] = np.array(pert_cat)
    results[cs.RES_DOSE_CAT] = np.array(dose_cat)
    results[cs.RES_CELL_TYPE] = np.array(cell_types)
    results[cs.RES_PRED] = torch.cat(pred).numpy()
    results[cs.RES_TRUTH] = torch.cat(truth).numpy()
    results[cs.RES_EXP_BATCH] = np.array(experimental_batches)

    return results
