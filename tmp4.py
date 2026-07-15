def longest_homopolymer(sequence):
    # Use regex to find all homopolymer stretches (e.g., A+, C+, G+, T+)
    homopolymers = re.findall(r"(A+|C+|G+|T+)", sequence)

    if homopolymers:
        # Find the length of the longest homopolymer
        max_length = len(max(homopolymers, key=len))

        # Collect all homopolymers that have the same length as the longest
        longest_homopolymers = [h for h in homopolymers if len(h) == max_length]

        # If there is only one longest homopolymer, return it as a string
        if len(longest_homopolymers) == 1:
            return max_length, longest_homopolymers[0]
        # If there are multiple longest homopolymers, return them as a list
        else:
            return max_length, sorted(list(set(longest_homopolymers)))
    else:
        return 0, None  # If no homopolymer is found

def triplet_stats(sequence):
    # Create a list of 3-mers (triplets) from the sequence
    triplets = [sequence[i : (i + 3)] for i in range(len(sequence) - 2)]

    # Number of distinct triplets
    distinct_triplets = set(triplets)

    # Number of total triplets
    total_triplets = len(triplets)

    # Triplet complexity: ratio of distinct triplets to total triplets
    triplet_complexity = len(distinct_triplets) / total_triplets if total_triplets > 0 else 0

    return len(distinct_triplets), total_triplets, triplet_complexity
