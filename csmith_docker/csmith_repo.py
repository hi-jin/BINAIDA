import requests
import re
from typing import Optional


class CsmithRepo:
    @classmethod
    def fetch_from_server(
        cls,
        server_url: str,
        max_funcs: int = 5,
        max_block_depth: int = 3,
    ) -> Optional[str]:
        params = {"max_funcs": max_funcs, "max_block_depth": max_block_depth}
        response = requests.get(server_url, params=params)

        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to retrieve data. HTTP Status code: {response.status_code}")
            return None

    @classmethod
    def parse_response(cls, content: str):
        c_code = []
        ir_code = []
        is_ir = False

        for line in content.splitlines():
            if line.startswith("Generated C code:"):
                is_ir = False
            elif line.startswith("Generated LLVM IR:"):
                is_ir = True
            elif not line.startswith("Content-Type:") and not line.startswith("HTTP/1.1"):
                if is_ir:
                    ir_code.append(line)
                else:
                    c_code.append(line)

        return c_code, ir_code

    @classmethod
    def clean_ir_code(cls, ir_code: str):
        cleaned_ir_code = []
        for line in ir_code:
            # Remove inline comments
            line = re.sub(r";.*", "", line)
            if line.strip():  # Remove empty lines
                cleaned_ir_code.append(line)
        return cleaned_ir_code

    @classmethod
    def save_to_file(cls, content: str, filename: str):
        with open(filename, "w") as file:
            file.write(content)


def main():
    server_url = "http://localhost:8080"
    max_funcs = 5
    max_block_depth = 3

    content = CsmithRepo.fetch_from_server(server_url, max_funcs, max_block_depth)
    if content:
        c_code, ir_code = CsmithRepo.parse_response(content)
        cleaned_ir_code = CsmithRepo.clean_ir_code(ir_code)

        CsmithRepo.save_to_file("\n".join(c_code), "random.c")
        CsmithRepo.save_to_file("\n".join(cleaned_ir_code), "random.ll")
        print("C code and LLVM IR have been saved to random.c and random.ll respectively.")
    else:
        print("Failed to get C code and LLVM IR.")


if __name__ == "__main__":
    main()
