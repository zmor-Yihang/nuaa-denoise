import subprocess
import os

def run_script(script_name):
    print(f"Running {script_name}...")
    subprocess.run(["python", script_name], check=True)
    print(f"{script_name} completed successfully.")

if __name__ == "__main__":
    # 询问用户是否要进行VMD参数优化
    optimize_vmd = input("是否要进行VMD参数优化? (y/n): ")
    
    if optimize_vmd.lower() == 'y':
        print("\n执行VMD参数优化流程...")
        run_script("vmd_parameter_optimization.py")
        
        # 询问用户是否要使用优化后的参数进行后续处理
        use_optimized = input("\n是否使用优化后的参数进行后续处理? (y/n): ")
        
        if use_optimized.lower() == 'y':
            print("\n使用优化参数执行标准流程...")
            # 使用优化参数进行VMD分解
            run_script("vmd_with_optimal_params.py")
            # 执行后续处理
            scripts = [
                "imf_feature_extraction.py",
                "imf_clustering.py",
                "signal_reconstruction.py"
            ]
        else:
            print("\n执行标准流程...")
            scripts = [
                "vmd_saveImf.py",
                "imf_feature_extraction.py",
                "imf_clustering.py",
                "signal_reconstruction.py"
            ]
    else:
        print("\n执行标准流程...")
        scripts = [
            "vmd_saveImf.py",
            "imf_feature_extraction.py",
            "imf_clustering.py",
            "signal_reconstruction.py"
        ]
    
    for script in scripts:
        run_script(script)
