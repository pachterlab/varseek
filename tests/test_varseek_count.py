import pytest
import sys
import importlib


def test_varseek_count_on_command_line(monkeypatch):
    import varseek.main as main_module

    test_args = [
        "vk", "count", 
        "-i", "shamindex.idx",
        "-g", "shamt2g.txt",
        "-x", "bulk",
        "--k", "11",
        "--adata_reference_genome", "ref_genome.fasta",
        "--kallisto", "shamkallisto",
        "--multiplexed",
        "myfastq.fastq"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    monkeypatch.setenv("TESTING", "true")  # so that main returns params_dict (there is a conditional in main to look for this environment variable)
    
    importlib.reload(main_module)
    fastqs, params_dict = main_module.main()  # now returns params_dict because TESTING is set

    expected_fastqs = ['myfastq.fastq']

    expected_dict = {
        'index': 'shamindex.idx',
        't2g': 'shamt2g.txt',
        'technology': 'bulk',
        'k': 11,
        'adata_reference_genome': 'ref_genome.fasta',
        'kallisto': 'shamkallisto',
        'multiplexed': True
    }
    
    assert fastqs == expected_fastqs
    assert params_dict == expected_dict