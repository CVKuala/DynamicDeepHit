import torch

def negative_log_likelihood(outcomes, cif, t, e, denominator):
    """
        Compute the log likelihood loss 
        This function is used to compute the survival loss
    """
    #print(f"e : {e}")
    #print(f"t : {t}")
    #print(f"denominator : {denominator}")
    loss, censored_cif = 0, 0
    for k, ok in enumerate(outcomes):
        # Censored cif
        #print(f"outcome number - {k}")
        censored_cif += cif[k][e == 0][torch.arange((e == 0).sum()), t[e == 0]]
        #print(f"censored : {censored_cif}")
        #print(f"intermediate cif : {cif[k][e==0]}")
        #print(f"intermediate cif shape: {cif[k][e==0].shape}")
        # Uncensored
        selection = e == (k + 1)
        loss += torch.sum(torch.log(ok[selection][torch.arange((selection).sum()), t[selection]]/denominator))
        #print(f"loss : {loss}")

    # Censored loss
    #print(f"loss 2 : {torch.sum(torch.log(torch.clamp(1 - (censored_cif/denominator),1e-5,1)))}")
    loss += torch.sum(torch.log(torch.clamp(1 - censored_cif,1e-5,1)))
    return - loss / len(outcomes)

def ranking_loss(cif, t, e, sigma):
    """
        Penalize wrong ordering of probability
        Equivalent to a C Index
        This function is used to penalize wrong ordering in the survival prediction
    """
    loss = 0
    # Data ordered by time
    for k, cifk in enumerate(cif):
        for ci, ti in zip(cifk[e-1 == k], t[e-1 == k]):
            # For all events: all patients that didn't experience event before
            # must have a lower risk for that cause
            if torch.sum(t > ti) > 0:
                # TODO: When data are sorted in time -> wan we make it even faster ?
                loss += torch.mean(torch.exp((cifk[t > ti][torch.arange((t > ti).sum()), ti] - ci[ti])) / sigma)

    return loss / len(cif)

def longitudinal_loss(longitudinal_prediction, x):
    """
        Penalize error in the longitudinal predictions
        This function is used to compute the error made by the RNN

        NB: In the paper, they seem to use different losses for continuous and categorical
        But this was not reflected in the code associated (therefore we compute MSE for all)

        NB: Original paper mentions possibility of different alphas for each risk
        But take same for all (for ranking loss)
    """
    length = (~torch.isnan(x[:,:,0])).sum(axis = 1) - 1
    if x.is_cuda:
        device = x.get_device()
    else:
        device = torch.device("cpu")

    # Create a grid of the column index
    index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)

    # Select all predictions until the last observed
    prediction_mask = index <= (length - 1).unsqueeze(1).repeat(1, x.size(1))

    # Select all observations that can be predicted
    observation_mask = index <= length.unsqueeze(1).repeat(1, x.size(1))
    observation_mask[:, 0] = False # Remove first observation

    return torch.nn.MSELoss(reduction = 'mean')(longitudinal_prediction[prediction_mask], x[observation_mask])

def total_loss(model, x, t, e, alpha, beta, sigma):
    longitudinal_prediction, outcomes = model(x)
    t, e = t.long(), e.int()
    #print(longitudinal_prediction.shape)
    #print(len(outcomes))
    #print(f"outcomes : {outcomes}")
    #print(outcomes[0].shape)
    #print(f"e : {e}")
    #print(f"t : {t}")
    outcomes_concat = torch.stack(outcomes, dim=0)  # Stack along a new dimension

    # Compute cumulative function from prediced outcomes
    cif = [torch.cumsum(ok, 1) for ok in outcomes]
    #print(f"cif : {cif}")

    tensor_data = torch.flip(torch.cumsum(torch.flip(outcomes_concat, [2]), 2), [2])[:,:,1:]

    zeros_column = torch.zeros(tensor_data.shape[0], tensor_data.shape[1], 1).cuda()  # Create a column of zeros
    result = torch.cat((tensor_data, zeros_column), dim=2)  # Concatenate along the last dimension

    denom = 1 - (result[0] + result[1])
    #print(f"Denominator : {denom}")

    cif = [each_out/(denom) for each_out in cif]
    #print(f"cif : {cif}")
    #print(f"maximum cif : {torch.max(cif[0])}")
    #print(f"minimum cif : {torch.min(cif[0])}")
    #print(f"maximum cif : {torch.max(cif[1])}")
    #print(f"minimum cif : {torch.min(cif[1])}")


    denominator = 0
    for k, ok in enumerate(outcomes):
        # Uncensored
        #selection = e == (k + 1)
        selection = torch.full(e.shape, True, dtype=torch.bool)
        #print(f"selection : {selection}")
        #print(f"t[selection] : {t[selection]}")
        temp_denominator = 0
        for each_selection in torch.arange((selection).sum()):
            temp_denominator += torch.sum(ok[selection][each_selection, t[selection][each_selection]:])
        temp_denominator = temp_denominator/(len(torch.arange((selection).sum())) + 1e-10)
        denominator+=temp_denominator
            
    #denominator = denominator/len(outcomes)
    denominator = 1 - denominator

    long_loss = longitudinal_loss(longitudinal_prediction, x)
    rank_loss = ranking_loss(cif, t, e, sigma)
    nll_loss = negative_log_likelihood(outcomes, cif, t, e, denominator)

    return (1 - alpha - beta) * long_loss + alpha * rank_loss + beta * nll_loss