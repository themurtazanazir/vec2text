import torch

def optimize_transform(llama_unembed, other_logprobs, llama_overlap_toks, other_overlap_toks):


    W = llama_unembed[llama_overlap_toks]
    B = other_logprobs.T[other_overlap_toks]
    B = B - B.mean(axis=0, keepdims=True)

    llama2_hidden_state, *_ = torch.linalg.lstsq(
            W.float(),B.float() )

    return llama2_hidden_state.T

def optimize_transform_alr(llama_unembed, other_logprobs, llama_overlap_toks, other_overlap_toks):


    W = llama_unembed[llama_overlap_toks]
    B = other_logprobs.T[other_overlap_toks]
    #B = B - B.mean(axis=0, keepdims=True)
    B = B - B[0]#.mean(axis=0, keepdims=True)

    llama2_hidden_state, *_ = torch.linalg.lstsq(
            W.float(),B.float() )

    return llama2_hidden_state.T

def optimize_transform_matt(llama_unembed, other_logprobs, llama_overlap_toks, other_overlap_toks):
    llama_unembed_and_alr = llama_unembed - llama_unembed[[llama_overlap_toks[0]]]
    other_output = other_logprobs - other_logprobs[:, [other_overlap_toks[0]]]
    llama2_hidden_state, *_ = torch.linalg.lstsq(
            llama_unembed_and_alr[llama_overlap_toks[1:]],
            other_output.T[other_overlap_toks[1:]]
            )

    return llama2_hidden_state.T


def optimize_transform_probs(llama_unembed, other_logprobs, llama_overlap_toks, other_overlap_toks):
    p = other_logprobs.exp()[:, other_overlap_toks]

    def f(x):
        return torch.sum(-p @ torch.nn.functional.log_softmax(llama_unembed@x.T, dim=-1)[llama_overlap_toks])

    x = torch.ones(16, llama_unembed.shape[1]).to(llama_unembed.device)
    x.requires_grad = True
    def closure():
        lbfgs.zero_grad()
        objective = f(x)
        objective.backward()
        return objective

    from torch import optim
    lbfgs = optim.LBFGS([x],
                        #history_size=10,
                        max_iter=4,
                        #line_search_fn="strong_wolfe",
                        )

    history_lbfgs = []
    for i in range(100):
        history_lbfgs.append(f(x).item())
        lbfgs.step(closure)
    print(history_lbfgs)

    topk_vals = torch.topk(torch.nn.functional.softmax(llama_unembed@x[0]), 10)
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    print(t.batch_decode(topk_vals.indices))
    return x
    
