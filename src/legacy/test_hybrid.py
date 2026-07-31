from retriever import (
    search_course,
    search_program,
    search_department,
    search_vector
)

question = input(
    "Ask a question: "
)

# =====================================
# COURSE SEARCH
# =====================================

result = search_course(question)

if result:

    print("\nCOURSE FOUND")
    print("-" * 50)

    print("Code:", result[0])
    print("Name:", result[1])
    print("Credits:", result[2])
    print("Department:", result[4])

    print("\nDescription:\n")
    print(result[3])

    exit()

# =====================================
# PROGRAM SEARCH
# =====================================

result = search_program(question)

if result:

    print("\nPROGRAM FOUND")
    print("-" * 50)

    print("Program:")
    print(result[0])

    print("\nAdmission Requirements:\n")
    print(result[1][:1000])

    print("\nProgram Requirements:\n")
    print(result[2][:1000])

    exit()

# =====================================
# DEPARTMENT SEARCH
# =====================================

result = search_department(question)

if result:

    print("\nDEPARTMENT FOUND")
    print("-" * 50)

    print("Department:")
    print(result[0])

    print("\nPrograms:\n")
    print(result[1])

    print("\nDescription:\n")
    print(result[2][:1000])

    exit()

# =====================================
# CHROMADB FALLBACK
# =====================================

print("\nNo database match found.")
print("Searching ChromaDB...\n")

results = search_vector(question)

for i in range(
    len(results["documents"][0])
):

    print(f"\nResult {i+1}")
    print("-" * 50)

    print("Chunk:\n")
    print(
        results["documents"][0][i][:1000]
    )

    print("\nSource:")
    print(
        results["metadatas"][0][i]["url"]
    )

    print()