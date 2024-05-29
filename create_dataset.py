import os
from csmith_docker.csmith_repo import CsmithRepo


def create_dataset(
    dataset_name: str,
    dataset_size: int,
    max_funcs: int,
    max_block_depth: int,
    output_dir: str,
    server_url: str = "http://localhost:8080",
):
    ##### create directory if it does not exist #####
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ##### create dataset #####
    repo = CsmithRepo
    for i in range(dataset_size):
        response = repo.fetch_from_server(
            server_url,
            max_funcs,
            max_block_depth,
        )
        if response is None:
            i -= 1
            print(f"Failed to fetch data for iteration {i}, retrying...")
            continue

        c_code, ir_code = repo.parse_response(response)
        cleaned_ir_code = repo.clean_ir_code(ir_code)

        c_code = "\n".join(c_code)
        ir_code = "\n".join(cleaned_ir_code)

        repo.save_to_file(c_code, f"{output_dir}/{dataset_name}_{i}.c")
        repo.save_to_file(ir_code, f"{output_dir}/{dataset_name}_{i}.ll")

        print(
            f"Saved C code and LLVM IR to {output_dir}/{dataset_name}_{i}.c and {output_dir}/{dataset_name}_{i}.ll respectively."
        )


if __name__ == "__main__":
    create_dataset(
        dataset_name="csmith_dataset",
        dataset_size=10,
        max_funcs=5,
        max_block_depth=3,
        output_dir="dataset",
    )
