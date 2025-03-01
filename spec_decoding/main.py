
import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2
from globals import Decoder

# Import QSPEC modules if available
try:
    from sampling import QuantizedModel, QSPECModel, qspec_sampling, qspec_sampling_different_models
    HAS_QSPEC = True
except ImportError:
    HAS_QSPEC = False




# my local models
MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    "llama2-7b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-7b-hf",
    "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    "bloom7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default="gpt2")
    parser.add_argument('--target_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')
    
    # QSPEC specific arguments
    if HAS_QSPEC:
        parser.add_argument('--use_qspec', action='store_true', default=False, help='use QSPEC sampling')
        parser.add_argument('--draft_quantization', type=str, default="fp8", choices=["none", "int8", "int4", "fp8", "fp4"], help='quantization method for draft model')
        parser.add_argument('--verify_quantization', type=str, default="int8", choices=["none", "int8", "int4", "fp8", "fp4"], help='quantization method for verification model')
        parser.add_argument('--same_model_qspec', action='store_true', default=False, help='use the same model for both draft and verification in QSPEC')
        parser.add_argument('--qspec_model_name', type=str, default=None, help='model name to use for both draft and verification in QSPEC (when same_model_qspec is True)')
        parser.add_argument('--qspec_draft_model_name', type=str, default=None, help='model name to use for draft in QSPEC (when same_model_qspec is False)')
        parser.add_argument('--qspec_verify_model_name', type=str, default=None, help='model name to use for verification in QSPEC (when same_model_qspec is False)')
    args = parser.parse_args()
    return args


def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 10
    profile_filename = f"./profile_logs/{print_prefix}"
    
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")

def generate(input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False,
             use_qspec = False, draft_quantization = "fp8", verify_quantization = "int8",
             same_model_qspec = False, qspec_model_name = None, qspec_draft_model_name = None, qspec_verify_model_name = None):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       trust_remote_code=True).to(torch_device)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       trust_remote_code=True).to(torch_device)
    print("finish loading models")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_large", use_profiling,
                  input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_small", use_profiling,
                  input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    
    torch.manual_seed(123)
    output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"deepmind's speculative_sampling: {generated_text}")   

    torch.manual_seed(123)
    output = speculative_sampling(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"google's speculative_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(speculative_sampling, "SP", use_profiling,
                  input_ids, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed)
    
    # QSPEC sampling if enabled
    if HAS_QSPEC and use_qspec:
        print("\n" + "="*50)
        print("Running QSPEC sampling...")
        print("="*50)
        
        if same_model_qspec:
            # Use the same model with different quantization schemes
            model_name = qspec_model_name or target_model_name
            print(f"Using same model ({model_name}) with different quantization schemes:")
            print(f"  - Draft: {draft_quantization}")
            print(f"  - Verification: {verify_quantization}")
            
            torch.manual_seed(123)
            output = qspec_sampling(
                input_ids,
                model_name,
                draft_quantization=draft_quantization,
                verify_quantization=verify_quantization,
                max_len=num_tokens,
                gamma=gamma,
                top_k=top_k,
                top_p=top_p,
                verbose=verbose,
                random_seed=random_seed,
                device=torch_device
            )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            color_print(f"QSPEC sampling (same model): {generated_text}")
            
            if use_benchmark:
                benchmark(qspec_sampling, "QSPEC_same_model", use_profiling,
                          input_ids, model_name, draft_quantization=draft_quantization,
                          verify_quantization=verify_quantization, max_len=num_tokens,
                          gamma=gamma, top_k=top_k, top_p=top_p, random_seed=random_seed,
                          device=torch_device)
        else:
            # Use different models with quantization
            draft_model = qspec_draft_model_name or approx_model_name
            verify_model = qspec_verify_model_name or target_model_name
            
            print(f"Using different models with quantization:")
            print(f"  - Draft: {draft_model} ({draft_quantization})")
            print(f"  - Verification: {verify_model} ({verify_quantization})")
            
            torch.manual_seed(123)
            output = qspec_sampling_different_models(
                input_ids,
                draft_model,
                verify_model,
                draft_quantization=draft_quantization,
                verify_quantization=verify_quantization,
                max_len=num_tokens,
                gamma=gamma,
                top_k=top_k,
                top_p=top_p,
                verbose=verbose,
                random_seed=random_seed,
                device=torch_device
            )
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            color_print(f"QSPEC sampling (different models): {generated_text}")
            
            if use_benchmark:
                benchmark(qspec_sampling_different_models, "QSPEC_diff_models", use_profiling,
                          input_ids, approx_model_name, target_model_name,
                          draft_quantization=draft_quantization, verify_quantization=verify_quantization,
                          max_len=num_tokens, gamma=gamma, top_k=top_k, top_p=top_p,
                          random_seed=random_seed, device=torch_device)

if __name__ == "__main__":
    args = parse_arguments()
    
    # Check if QSPEC is available and requested
    use_qspec = False
    draft_quantization = "fp8"
    verify_quantization = "int8"
    same_model_qspec = False
    qspec_model_name = None
    qspec_draft_model_name = None
    qspec_verify_model_name = None
    
    if HAS_QSPEC and hasattr(args, 'use_qspec'):
        use_qspec = args.use_qspec
        draft_quantization = args.draft_quantization
        verify_quantization = args.verify_quantization
        same_model_qspec = args.same_model_qspec
        qspec_model_name = args.qspec_model_name
        qspec_draft_model_name = args.qspec_draft_model_name
        qspec_verify_model_name = args.qspec_verify_model_name
    
    generate(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
             random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark, use_profiling = args.profiling,
             use_qspec = use_qspec, draft_quantization = draft_quantization, verify_quantization = verify_quantization,
             same_model_qspec = same_model_qspec, qspec_model_name = qspec_model_name,
             qspec_draft_model_name = qspec_draft_model_name, qspec_verify_model_name = qspec_verify_model_name)
