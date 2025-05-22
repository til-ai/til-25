# docker_verify.py
import os
# import paddle # Defer this import

print('--- Docker Build Sanity Check ---')
PADDLE_IMPORT_SUCCESS = False
PADDLE_VERSION = "N/A"
CUDA_COMPILED = "N/A"

try:
    import paddle # Attempt import here
    PADDLE_IMPORT_SUCCESS = True
    PADDLE_VERSION = paddle.__version__
    CUDA_COMPILED = paddle.is_compiled_with_cuda()
    print('Successfully imported PaddlePaddle.')
    print('Paddle version:', PADDLE_VERSION)
    print('CUDA available (compiled):', CUDA_COMPILED)

    if CUDA_COMPILED:
        try:
            gpu_count = paddle.device.cuda.device_count() # This might also fail if libcuda.so.1 is needed early
            print(f"Number of GPUs detected by PaddlePaddle (may be 0 during build): {gpu_count}")
            if gpu_count > 0: # Unlikely to be > 0 during build
                place = paddle.CUDAPlace(0)
                tensor = paddle.rand([2, 3], place=place)
                print(f'Test tensor on GPU OK. Tensor: {tensor.numpy()}')
            else:
                print("PaddlePaddle compiled with CUDA, but no GPUs actively detected/usable by Paddle (expected during build).")
        except Exception as e:
            print(f'GPU specific check FAILED (this is often expected during "docker build"): {e}')
            if "libcuda.so.1" in str(e):
                print("This error regarding libcuda.so.1 is common during 'docker build' as GPUs/drivers are not available to the build context.")
            print("The real GPU functionality test happens at 'docker run --gpus ...'")
    else:
        print('Paddle not compiled with CUDA. GPU will not be used.')

except ImportError as ie:
    if "libcuda.so.1" in str(ie):
        print(f"WARNING: Failed to import paddle due to missing 'libcuda.so.1' (ImportError: {ie}).")
        print("This is EXPECTED during 'docker build' as host GPU drivers are not available to the build environment.")
        print("PaddlePaddle GPU functionality will be tested at container runtime with --gpus flag.")
    else:
        print(f"CRITICAL: Failed to import paddle for a non-GPU related reason. Error: {ie}")
        print("This likely means the paddlepaddle-gpu installation failed or is corrupted.")
except Exception as e:
    print(f"An unexpected error occurred during PaddlePaddle import or checks: {e}")


print('\n--- Checking Model Paths (from ENV VARS) ---')
models_base_dir_env = os.getenv('MODELS_BASE_DIR')
layout_model_subdir_env = os.getenv('LAYOUT_MODEL_SUBDIR') # Just checking one for brevity

if models_base_dir_env and layout_model_subdir_env:
    layout_base = os.path.join(models_base_dir_env, layout_model_subdir_env)
    print(f"Expecting layout model base at: {layout_base}")
    if os.path.exists(layout_base):
        print(f'Layout model base exists at: {layout_base}')
        if os.path.isdir(layout_base):
            try:
                print(f'Contents of {layout_base}: {os.listdir(layout_base)}')
            except Exception as e:
                print(f"Could not list contents of {layout_base}: {e}")
        else:
            print(f'{layout_base} is not a directory.')
    else:
        print(f'Layout model base NOT FOUND at: {layout_base}')
else:
    print("MODELS_BASE_DIR or LAYOUT_MODEL_SUBDIR environment variables not set for model path check.")

if not PADDLE_IMPORT_SUCCESS and "libcuda.so.1" in globals().get("ie_str", "") : # Check if import failed due to libcuda
     print("\nVERIFICATION SUMMARY: Paddle import skipped due to expected 'libcuda.so.1' issue during build. Model paths checked.")
     print("Final GPU functionality test occurs at runtime.")
elif not PADDLE_IMPORT_SUCCESS:
     print("\nVERIFICATION SUMMARY: CRITICAL - Paddle import failed for other reasons. Review installation.")
     # Consider exiting with an error code if import fails for non-libcuda reasons
     # import sys
     # sys.exit(1)
else:
     print("\nVERIFICATION SUMMARY: Paddle imported. Model paths checked.")


print('--- Sanity Check Done ---')

# To ensure the script doesn't cause the build to fail if paddle import fails due to libcuda.so.1
# we won't explicitly exit with error code 1 for that specific case.
# If import fails for other reasons, one might consider exiting with error to halt the build.