import os
import subprocess
import pysam
samtools = "/home/jrich/miniconda3/envs/kvar/bin/samtools"
input_bam_list = ['/home/jrich/data/varseek_data/trash/gatk_nov17/alignment/merged.bam']
threads = 8

for input_bam in input_bam_list:
    bam_file_name = input_bam.split("/")[-1]
    output_dir = os.path.dirname(input_bam)

    # Extract BAM header
    header = pysam.samtools.view('-H', input_bam)  #? samtools view -H INPUT.bam | grep "@RG"
    read_group_lines = [line for line in header.splitlines() if "@RG" in line]

    if len(read_group_lines) > 1:
        # Split BAM by read group
        format_string = output_dir + "/%*_%#.%."
        split_command = f"{samtools} split -f {format_string} {input_bam}"
        subprocess.run(split_command, shell=True)
        # _ = pysam.samtools.split('-f', format_string, "--threads", threads, input_bam)  #? samtools split -f $format_string INPUT.bam
        read_group_ids = []
        for line in read_group_lines:
            # Split the line by tabs and find the field starting with "ID:"
            fields = line.split("\t")
            for field in fields:
                if field.startswith("ID:"):
                    read_group_ids.append(field.split(":", 1)[1])  # Extract the value after "ID:"
        
        for rgid in read_group_ids:
            input_bam_rg = f"{output_dir}/{bam_file_name}_{rgid}.bam"
            output_fastq_rg = f"{output_dir}/{bam_file_name}_{rgid}.fastq.gz"
            command = f"{samtools} fastq -O {input_bam_rg} > {output_fastq_rg}"
            subprocess.run(command, shell=True)
            # with open(output_fastq_rg, "w") as output_file:
            #     pysam.samtools.fastq(
            #         "-O",  # Optimize output
            #         input_bam_rg,  # Input BAM file
            #         catch_stdout=False,  # Ensure output goes to the specified file
            #         stdout=output_file  # Redirect stdout to the file
            #     )  #? samtools fastq -O INPUT.bam > OUTPUT.fastq.gz
    else:
        output_fastq = f"{output_dir}/{bam_file_name}.fastq.gz"
        command = f"{samtools} fastq -O {input_bam} > {output_fastq}"
        subprocess.run(command, shell=True)
        # with open(output_fastq, "w") as output_file:
        #     pysam.samtools.fastq(
        #         "-O",  # Optimize output
        #         input_bam,  # Input BAM file
        #         catch_stdout=False,  # Ensure output goes to the specified file
        #         stdout=output_file  # Redirect stdout to the file
        #     )  #? samtools fastq -O INPUT.bam > OUTPUT.fastq.gz