#!/bin/bash

# Read HTTP request from stdin
read request

# Extract query string
query_string=$(echo $request | awk '{print $2}' | cut -d'?' -f2)

# Parse query parameters
max_funcs=$(echo $query_string | grep -oP '(?<=max_funcs=)[^&]*')
max_block_depth=$(echo $query_string | grep -oP '(?<=max_block_depth=)[^&]*')

# Set default values if parameters are not provided
max_funcs=${max_funcs:-5}
max_block_depth=${max_block_depth:-3}

# Generate a random C file using Csmith with length restrictions
csmith --max-funcs "$max_funcs" --max-block-depth "$max_block_depth" > /tmp/random.c

# Compile the C file to LLVM IR
clang -S -emit-llvm /tmp/random.c -o /tmp/random.ll

# Respond with the HTTP headers and content
{
  echo -e "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n"
  echo "Generated C code:"
  cat /tmp/random.c
  echo ""
  echo "Generated LLVM IR:"
  cat /tmp/random.ll
} 2>&1
