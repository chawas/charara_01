#!/usr/bin/env python3
with open('scripts/era5_analysis_01.py', 'r') as f:
    lines = f.readlines()
    
print(f"Total lines: {len(lines)}")
print("\nLine 59 (actual content):")
print(f"'{lines[58].rstrip()}'")  # Python uses 0-based indexing

print("\nContext (lines 55-65):")
for i in range(54, 65):
    line_num = i + 1
    content = lines[i].rstrip()
    print(f"{line_num:3}: {repr(content)}")

print("\nChecking for invisible characters:")
for i in range(54, 65):
    line_num = i + 1
    content = lines[i]
    if any(ord(c) > 127 or ord(c) < 32 and c not in '\t\n\r' for c in content):
        print(f"Line {line_num} has special characters:")
        for j, char in enumerate(content):
            if ord(char) > 127 or (ord(char) < 32 and char not in '\t\n\r'):
                print(f"  Position {j}: '{char}' (ASCII {ord(char)})")
