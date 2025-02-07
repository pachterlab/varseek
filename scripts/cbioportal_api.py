import colorsys
import json
import math
import os
import subprocess
from collections import OrderedDict, defaultdict
from itertools import islice
from typing import Literal, TypeAlias, TypeVar

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bravado.client import SwaggerClient
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from upsetplot import UpSet

from scripts.map_enst_to_ensg import (
    get_ensembl_gene_name_bulk,
    get_valid_ensembl_gene_id_bulk,
)

_K = TypeVar('_K')
_V = TypeVar('_V')

if not hasattr(pd.DataFrame, "map"):
    print("Old version of pandas detected. Patching DataFrame.map to DataFrame.applymap")
    pd.DataFrame.map = pd.DataFrame.applymap

cancer_type_to_tissue_dictionary = {
    'Acute Leukemias of Ambiguous Lineage': 'leukemia',
    'Acute Myeloid Leukemia': 'leukemia',
    'Acute myeloid leukemia': 'leukemia',
    'Adenosarcoma': 'mixed',
    'Adrenal Tumor': 'adrenal_gland',
    'Adrenocortical Adenoma': 'adrenal_gland',
    'Adrenocortical Carcinoma': 'adrenal_gland',
    'Adrenocortical carcinoma': 'adrenal_gland',
    'Ampullary Cancer': 'ampulla',
    'Ampullary Carcinoma': 'ampulla',
    'Anal Cancer': 'intestine',
    'Appendiceal Cancer': 'appendix',
    'B-Lymphoblastic Leukemia/Lymphoma': 'lymphoma',
    'Biliary Tract': 'biliary_tract',
    'Biliary Tract Cancer, NOS': 'biliary_tract',
    'Bladder Cancer': 'bladder',
    'Bladder/Urinary Tract Cancer, NOS': 'bladder',
    'Blastic Plasmacytoid Dendritic Cell Neoplasm': 'immune',
    'Blood Cancer, NOS': 'leukemia',
    'Bone Cancer': 'bone',
    'Bone Sarcoma': 'bone',
    'Bowel Cancer, NOS': 'intestine',
    'Brain Cancer': 'brain',
    'Breast Cancer': 'breast',
    'Breast Carcinoma': 'breast',
    'Breast Sarcoma': 'breast',
    'CNS Cancer': 'brain',
    'Cancer of Unknown Primary': 'mixed',
    'Carcinoma of Uterine Cervix': 'cervix',
    'Cervical Cancer': 'cervix',
    'Cholangiocarcinoma': 'biliary_tract',
    'Choroid Plexus Tumor': 'brain',
    'Colorectal Cancer': 'intestine',
    'Colorectal Carcinoma': 'intestine',
    'Cutaneous malignancy of hair matrix cells': 'skin',
    'Diffuse Glioma': 'brain',
    'Embryonal Tumor': 'mixed',
    'Encapsulated Glioma': 'brain',
    'Endometrial Cancer': 'uterus',
    'Endometrial Carcinoma': 'uterus',
    'Ependymomal Tumor': 'brain',
    'Esophageal Carcinoma': 'esophagus',
    'Esophagogastric Cancer': 'esophagus',
    'Essential Thrombocythemia': 'plasma',
    'Extrahepatic Cholangiocarcinoma': 'biliary_tract',
    'Fibrosarcoma': 'soft_tissue',
    'Gallbladder Carcinoma': 'gallbladder',
    'Gastric Cancer': 'stomach',
    'Gastrointestinal Neuroendocrine Tumor': 'intestine',
    'Gastrointestinal Stromal Tumor': 'intestine',
    'Germ Cell Tumor': 'testicle',
    'Gestational Trophoblastic Disease': 'uterus',
    'Glioblastoma': 'brain',
    'Glioma': 'brain',
    'Head and Neck Cancer': 'head_neck',
    'Head and Neck Cancer, NOS': 'head_neck',
    'Head and Neck Carcinoma': 'head_neck',
    'Hepatobiliary Cancer': 'liver',
    'High-grade glioma/astrocytoma': 'brain',
    'Histiocytosis': 'immune',
    'Hodgkin Lymphoma': 'lymphoma',
    'Hodgkin Lymphoma-like PTLD': 'lymphoma',
    'Intraductal Papillary Mucinous Neoplasm': 'pancreas',
    'Intrahepatic Cholangiocarcinoma': 'biliary_tract',
    'Invasive Breast Carcinoma': 'breast',
    'Kidney Renal Cell Carcinoma': 'kidney',
    'Leukemia': 'leukemia',
    'Liver Hepatocellular Carcinoma': 'liver',
    'Liver Tumor': 'liver',
    'Low-grade glioma/astrocytoma': 'brain',
    'Lung Adenocarcinoma': 'lung',
    'Lung Cancer': 'lung',
    'Lung Cancer, NOS': 'lung',
    'Lung cancer': 'lung',
    'Lymphoid Neoplasm': 'lymph',
    'Malignant Rhabdoid Tumor of the Liver': 'liver',
    'Mastocytosis': 'immune',
    'Mature B-Cell Neoplasms': 'lymphoma',
    'Mature B-cell lymphoma': 'lymphoma',
    'Mature T and NK Neoplasms': 'immune',
    'Medulloblastoma': 'brain',
    'Melanoma': 'skin',
    'Meningioma': 'brain',
    'Mesothelioma': 'soft_tissue',
    'Miscellaneous Brain Tumor': 'brain',
    'Miscellaneous Neuroepithelial Tumor': 'brain',
    'Mucinous Adenocarcinoma Lymph Node': 'lymph',
    'Myelodysplastic Syndromes': 'plasma',
    'Myelodysplastic/Myeloproliferative Neoplasms': 'plasma',
    'Myeloproliferative Neoplasms': 'plasma',
    'Nerve Sheath Tumor': 'soft_tissue',
    'Nested stromal epithelial tumor of the liver': 'liver',
    'Non Small Cell Lung Cancer': 'lung',
    'Non-Germinomatous Germ Cell Tumor': 'testicle',
    'Non-Hodgkin Lymphoma': 'lymphoma',
    'Non-Seminomatous Germ Cell Tumor': 'testicle',
    'Non-Small Cell Lung Cancer': 'lung',
    'Ocular Melanoma': 'eye',
    'Other': 'mixed',
    'Ovarian Cancer': 'ovary',
    'Ovarian Carcinoma': 'ovary',
    'Ovarian Epithelial Tumor': 'ovary',
    'Ovarian Germ Cell Tumor': 'ovary',
    'Ovarian/Fallopian Tube Cancer, NOS': 'ovary',
    'Pancreatic Cancer': 'pancreas',
    'Penile Cancer': 'intestine',
    'Peripheral Nervous System': 'soft_tissue',
    'Pheochromocytoma': 'adrenal_gland',
    'Pineal Tumor': 'brain',
    'Pleural Mesothelioma': 'soft_tissue',
    'Posttransplant Lymphoproliferative Disorders': 'lymphoma',
    'Prostate Cancer': 'prostate',
    'Prostate Cancer, NOS': 'prostate',
    'Renal Cell Carcinoma': 'kidney',
    'Renal Clear Cell Carcinoma': 'kidney',
    'Renal Non-Clear Cell Carcinoma': 'kidney',
    'Renal cancer': 'kidney',
    'Retinoblastoma': 'eye',
    'Rhabdoid Cancer': 'soft_tissue',
    'Salivary Cancer': 'head_neck',
    'Salivary Gland Cancer': 'head_neck',
    'Salivary Gland-Type Tumor of the Lung': 'lung',
    'Sarcoma': 'soft_tissue',
    'Sellar Tumor': 'brain',
    'Seminoma': 'testicle',
    'Sex Cord Stromal Tumor': 'testicle',
    'Skin Cancer, Non-Melanoma': 'skin',
    'Small Bowel Cancer': 'intestine',
    'Small Bowel Carcinoma': 'intestine',
    'Small Cell Lung Cancer': 'lung',
    'Soft Tissue Myoepithelial Carcinoma': 'soft_tissue',
    'Soft Tissue Sarcoma': 'soft_tissue',
    'Soft Tissue Tumor': 'soft_tissue',
    'T-Lymphoblastic Leukemia/Lymphoma': 'leukemia',
    'Teratoma with Malignant Transformation': 'testicle',
    'Thymic Epithelial Tumor': 'thymus',
    'Thymic Tumor': 'thymus',
    'Thyroid Cancer': 'thyroid',
    'Thyroid Carcinoma': 'thyroid',
    'Urothelial Carcinoma': 'bladder',
    'Uterine Corpus Endometrial Carcinoma': 'uterus',
    'Uterine Endometrioid Carcinoma': 'uterus',
    'Uterine Sarcoma': 'uterus',
    'Vaginal Cancer': 'intestine',
    'Wilms Tumor': 'kidney'
}


cancer_type_acronym_to_tissue_dictionary = {'acbc': 'breast',
 'acc': 'adrenal_gland',
 'acyc': 'adenoid',
 'aml': 'leukemia',
 'ampca': 'ampulla',
 'angs': 'endothelial',
 'apad': 'appendix',
 'bcc': 'skin',
 'bfn': 'breast',
 'biliary_tract': 'biliary_tract',
 'blca': 'bladder',
 'bll': 'leukemia',
 'bowel': 'intestine',
 'brain': 'brain',
 'brca': 'breast',
 'breast': 'breast',
 'ccrcc': 'kidney',
 'cervix': 'cervix',
 'cesc': 'cervix',
 'chol': 'biliary_tract',
 'chrcc': 'kidney',
 'cllsll': 'leukemia',
 'coad': 'intestine',
 'coadread': 'intestine',
 'cscc': 'skin',
 'desm': 'skin',
 'difg': 'brain',
 'dlbclnos': 'lymphoma',
 'egc': 'esophagus',
 'es': 'bone',
 'esca': 'esophagus',
 'escc': 'esophagus',
 'gbc': 'gallbladder',
 'gbm': 'brain',
 'gist': 'stomach',
 'hcc': 'liver',
 'hccihch': 'liver',
 'hdcn': 'immune',
 'head_neck': 'head_neck',
 'hgsoc': 'ovary',
 'hnsc': 'head_neck',
 'ihch': 'liver',
 'lgsoc': 'ovary',
 'liad': 'liver',
 'luad': 'lung',
 'lung': 'lung',
 'lusc': 'lung',
 'lymph': 'lymph',
 'mbc': 'breast',
 'mbl': 'brain',
 'mbn': 'lymphoma',
 'mcl': 'lymphoma',
 'mds': 'leukemia',
 'mel': 'skin',
 'mixed': 'mixed',
 'mng': 'head_neck',
 'mnm': 'leukemia',
 'mpn': 'leukemia',
 'mpnst': 'soft_tissue',
 'mrt': 'kidney',
 'mtnn': 'lymphoma',
 'myeloid': 'leukemia',
 'nbl': 'adrenal_gland',
 'nccrcc': 'kidney',
 'nhl': 'lymphoma',
 'npc': 'head_neck',
 'nsclc': 'lung',
 'nsgct': 'testicle',
 'nst': 'brain',
 'odg': 'brain',
 'ovary': 'ovary',
 'paac': 'pancreas',
 'paad': 'pancreas',
 'pact': 'pancreas',
 'pancreas': 'pancreas',
 'panet': 'pancreas',
 'past': 'brain',
 'pcm': 'plasma',
 'pcnsl': 'lymphoma',  # brain?
 'plmeso': 'lung',
 'prad': 'prostate',
 'prcc': 'kidney',
 'prostate': 'prostate',
 'rbl': 'eye',
 'rms': 'muscle',
 'scco': 'ovary',
 'sclc': 'lung',
 'skcm': 'skin',
 'soft_tissue': 'soft_tissue',
 'stad': 'stomach',
 'stmyec': 'soft_tissue',
 'stomach': 'stomach',
 'testis': 'testicle',
 'tet': 'thymus',
 'thpa': 'thyroid',
 'thym': 'thymus',
 'thyroid': 'thyroid',
 'uccc': 'uterus',
 'ucec': 'uterus',
 'ucs': 'uterus',
 'um': 'eye',
 'urcc': 'kidney',
 'usarc': 'uterus',
 'utuc': 'bladder',
 'vsc': 'cervix',
 'wt': 'kidney'}


class CbioPortalStudy:
    def __init__(self, study_id = None):
        self.cbioportal = SwaggerClient.from_url(
            'https://www.cbioportal.org/api/v2/api-docs',
            config={"validate_requests": False, "validate_responses": False, "validate_swagger_spec": False}
        )

        for a in dir(self.cbioportal):
            setattr(self.cbioportal, a.replace(' ', '_').lower(), getattr(self.cbioportal, a))

        self.studies = self.cbioportal.studies.getAllStudiesUsingGET().result()

        self.study_id = study_id

        self.cancer_type_acronym_to_tissue_dictionary = cancer_type_acronym_to_tissue_dictionary

    def list_attributes_keys(self):
        print("Attributes of CbioPortalStudy:")
        for key in vars(self).keys():
            print(key)

    def list_attributes(self):
        for key, value in vars(self).items():
            print(f"{key}: {value}")

    def _get_clinical_data(self):
        # Retrieve clinical data for the study
        self.clinical_data = self.cbioportal.clinical_data.getAllClinicalDataInStudyUsingGET(studyId=self.study_id).result()

        # todo what is going on here? should the {} be around the whole generator-y construct?
        self.cdf = pd.DataFrame.from_dict(
            {k:getattr(cd,k) for k in dir(cd)} for cd in self.clinical_data
        ).pivot(index='sampleId patientId'.split(),columns='clinicalAttributeId',values='value')

    def print_all_endpoints(self):
        endpoints = dir(self.cbioportal)
        print("Available Endpoints:")
        for endpoint in endpoints:
            print(endpoint)

    def print_all_study_ids(self):
        for entry in self.studies:
            print(f"Study ID **{entry.studyId}**: Cancer type **{entry.cancerTypeId}** ({entry.name}) - {entry.description}")

            # {self.extract_name(individual_study['name']): individual_study['cancerTypeId'] for individual_study in self.studies}

    def calculate_study_summary_info(self):
        patients = self.cbioportal.Patients.getAllPatientsInStudyUsingGET(studyId=self.study_id).result()
        self.total_patients = len(patients)

        # Retrieve mutation data for the study
        molecular_profile_id = f'{self.study_id}_mutations'
        sample_list_id = f'{self.study_id}_all'

        if not hasattr(self, 'all_mutations'):
            self.all_mutations = self.cbioportal.mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(
                molecularProfileId=molecular_profile_id,
                sampleListId=sample_list_id,
                projection="DETAILED"
            ).result()

        # Calculate the total number of mutations
        total_mutations = len(self.all_mutations)

        # Find all unique gene symbols in the mutations data
        unique_genes = set(mutation['gene']['hugoGeneSymbol'] for mutation in self.all_mutations)

        self.total_mutated_genes = len(unique_genes)

        study_summary = {'total_patients': self.total_patients, 'total_mutations': total_mutations, 'total_mutated_genes': self.total_mutated_genes}

        return study_summary

    def print_study_summary_info(self):
        print(f"The msk_impact_2017 study spans {self.total_patients} patients")
        print(f"Total number of recorded mutations: {self.total_patients}")
        print(f"Total number of unique genes with recorded mutations: {self.total_mutated_genes}")

    def get_cancer_type_information(self):
        # Filter data for 'CANCER_TYPE' and 'CANCER_TYPE_DETAILED'
        cancer_type_data = [data for data in self.clinical_data if data['clinicalAttributeId'] in ['CANCER_TYPE', 'CANCER_TYPE_DETAILED']]

        cancer_type_dict = {}
        for data in cancer_type_data:
            attribute_id = data['clinicalAttributeId']
            sample_id = data['sampleId']
            attribute_value = data['value']
            if attribute_id not in cancer_type_dict:
                cancer_type_dict[attribute_id] = {}
            cancer_type_dict[attribute_id][sample_id] = attribute_value

        # Find unique values for CANCER_TYPE and CANCER_TYPE_DETAILED
        self.unique_cancer_types = set(cancer_type_dict['CANCER_TYPE'].values()) if 'CANCER_TYPE' in cancer_type_dict else set()
        self.unique_cancer_type_detailed = set(cancer_type_dict['CANCER_TYPE_DETAILED'].values()) if 'CANCER_TYPE_DETAILED' in cancer_type_dict else set()

        # Set the lengths of unique values
        self.num_unique_cancer_types = len(self.unique_cancer_types)
        self.num_unique_cancer_type_detailed = len(self.unique_cancer_type_detailed)

    def print_cancer_type_information(self):
        print("Cancer Type Data:")
        for attribute_id, values in self.cancer_type_dict.items():
            print(f"\n{attribute_id} (first 5 values):")
            for sample_id, attribute_value in islice(values.items(), 5):
                print(f"Sample ID: {sample_id}, Value: {attribute_value}")
            print("...")

        print("\nTotal number of unique values in CANCER_TYPE:", self.num_unique_cancer_types)
        print("Total number of unique values in CANCER_TYPE_DETAILED:", self.num_unique_cancer_type_detailed)

        print("\nUnique values in CANCER_TYPE:")
        for value in self.unique_cancer_types:
            print(value)

        print("-----------------------------------------------------")

        print("\nUnique values in CANCER_TYPE_DETAILED:")
        for value in self.unique_cancer_type_detailed:
            print(value)

    def get_specific_cancer_information(self, cancer_types, cancer_type_category = 'CANCER_TYPE', cancer_types_name = ""):  # 'CANCER_TYPE' or 'CANCER_TYPE_DETAILED'
        if cancer_types_name == "":
            cancer_types_name = cancer_types[0]

        cancer_type_samples = [data['sampleId'] for data in self.clinical_data if data['clinicalAttributeId'] == cancer_type_category and data['value'] in cancer_types]

        self.cancer_type_info[cancer_type_category][cancer_types_name] = cancer_type_samples

        # Retrieve mutation data for all samples in the study
        molecular_profile_id = f'{self.study_id}_mutations'
        sample_list_id = f'{self.study_id}_all'

        if not hasattr(self, 'all_mutations'):
            self.all_mutations = self.cbioportal.mutations.getMutationsInMolecularProfileBySampleListIdUsingGET(
                molecularProfileId=molecular_profile_id,
                sampleListId=sample_list_id,
                projection="DETAILED"
            ).result()

        # Filter mutations for the specific samples
        if cancer_type_category == 'CANCER_TYPE':
            self.cancer_type_mutations = [mutation for mutation in self.all_mutations if mutation['sampleId'] in cancer_type_samples]
            cancer_type_mutations = self.cancer_type_mutations

            self.cancer_type_mutated_genes = list(set(mutation['gene']['hugoGeneSymbol'] for mutation in cancer_type_mutations))
            cancer_type_mutated_genes = self.cancer_type_mutated_genes
        elif cancer_type_category == 'CANCER_TYPE_DETAILED':
            self.cancer_type_mutations_detailed = [mutation for mutation in self.all_mutations if mutation['sampleId'] in cancer_type_samples]
            cancer_type_mutations = self.cancer_type_mutations_detailed

            self.cancer_type_mutated_genes_detailed = list(set(mutation['gene']['hugoGeneSymbol'] for mutation in cancer_type_mutations))
            cancer_type_mutated_genes = self.cancer_type_mutated_genes_detailed

        # Print some of the mutation data for brevity
        print(f"Total mutations found for {cancer_types_name} samples: {len(cancer_type_mutations)}")
        print("Example mutation data:")
        for mutation in cancer_type_mutations[:5]:  # Print first 5 mutations
            print(mutation)

        print(f"Total unique genes in {cancer_types_name} samples: {len(cancer_type_mutated_genes)}")
        print("Example gene data:")
        for gene in cancer_type_mutated_genes[:5]:  # Print first 5 mutations
            print(gene)

    def _retrieve_study_metadata(self):
        matching_study = None
        for row in self.studies:
            if row.studyId == self.study_id:
                matching_study = row
                break

        return matching_study

    def extract_name(self, name):
        parenthesis_index = name.find('(')
        if parenthesis_index != -1:
            return name[:parenthesis_index].strip()
        return name

    def make_cancer_type_acronym_dictionary(self):
        self.cancer_type_acronym_dictionary = {self.extract_name(individual_study['name']): individual_study['cancerTypeId'] for individual_study in self.studies}
        self.cancer_type_acronym_dictionary = OrderedDict(sorted(self.cancer_type_acronym_dictionary.items()))

    def find_study_row(self):
        for study in self.studies:
            if study.get('study_id') == self.study_id:
                return study
        return None

    @property
    def study_id(self):
        """Getter for study_id"""
        return self._study_id

    @study_id.setter
    def study_id(self, value):
        """Setter for study_id"""
        self._study_id = value

        if self._study_id:

            self._get_clinical_data()

            self.study_metadata = self._retrieve_study_metadata()

            self.cancer_type_id = self.study_metadata['cancerTypeId']

            self.description = self.study_metadata['description']

            self.calculate_study_summary_info()

            if self.cancer_type_id == "mixed":

                self.unique_cancer_types = set()
                self.unique_cancer_type_detailed = set()
                self.num_unique_cancer_types = 0
                self.num_unique_cancer_type_detailed = 0

                self.cancer_type_info = defaultdict(lambda: defaultdict(list))

                self.get_cancer_type_information()

    def find_study_ids_by_keyword(self, key_words):
        if not hasattr(self, 'cancer_type_acronym_dictionary'):
            self.make_cancer_type_acronym_dictionary()

        self.cancer_id_list = [
            cancer_type_acronym
            for cancer_type, cancer_type_acronym in self.cancer_type_acronym_dictionary.items()
            if any(key_word in cancer_type.lower() or key_word in cancer_type_acronym.lower() for key_word in key_words)
            and cancer_type_acronym.lower() != "mixed"
        ]

        self.matching_study_ids = [individual_study['studyId'] for individual_study in self.studies if individual_study['cancerTypeId'] in self.cancer_id_list]

        return self.matching_study_ids

    def get_cna_info(self, molecular_profile_id = None, sample_list_id = None):
        if not molecular_profile_id:
            molecular_profile_id = f'{self.study_id}_cna'

        if not sample_list_id:
            sample_list_id = f'{self.study_id}_all'

        self.cna_info = self.cbioportal.Discrete_Copy_Number_Alterations.getDiscreteCopyNumbersInMolecularProfileUsingGET(
            molecularProfileId=molecular_profile_id,
            sampleListId=sample_list_id,
            projection="DETAILED"
        ).result()

        return self.cna_info

    def get_sv_info(self, molecular_profile_id = None, sample_list_id = None):
        if not molecular_profile_id:
            molecular_profile_id = f'{self.study_id}_structural_variants'  # this is likely the right id suffix

        if not sample_list_id:
            sample_list_id = f'{self.study_id}_all'

        self.sv_info = ""  #TODO: Implement this


def nested_defaultdict() -> defaultdict[_K, _V | defaultdict[_K]]:
    return defaultdict(nested_defaultdict)


def convert_default_dict_to_standard_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_default_dict_to_standard_dict(v) for k, v in d.items()}
    return d


def ints_between(start: int, end: int, max_count: int, min_count: int, verbose: bool = False) -> list[int]:
    """
    Generate a list of integers between start and end (inclusive) with a maximum count of max_count and a minimum count min_count.
    The list is guaranteed to contain start and end, and the spacing between the numbers will be as even as possible.
    If a perfect spacing is not possible, the spacing will omit a number rather than overcrowding.

    This method is designed to be used for plot labels
    """
    assert max_count >= 2, "max_count must be at least 2"
    assert start < end, "start must be less than end"

    if max_count == 2:
        return [start, end]
    else:
        step = int(math.ceil((end - start) / (max_count - 1)))
        if verbose:
            print(f"Original step: {step}, {(end - start) % step=}")
            print(f"{end - start=}")
        # check if it comes out even
        while ((end - start) / step) + 1 >= min_count and (end - start) % step != 0:
            step += 1
            if verbose:
                print(f"New step: {step}, {(end - start) % step=}")
        if (end - start) % step != 0:
            step = int(math.ceil((end - start) / (max_count - 1)))
            if verbose:
                print("Reverted step")

        out = [start]
        current = start
        while current + step*2 <= end:
            current += step
            out.append(current)
        out.append(end)
        if verbose:
            print(f"{out=}")
        return out


class GeneAnalysis:
    def __init__(
            self,
            study_ids: str | list[str],
            cbioportal_data_out_dir: str = ".",
            merge_on_gene_name_type: Literal["Symbol", "Ensembl"] = "Symbol",
            merge_on_gene_name_type_upset: Literal["Symbol", "Ensembl"] = "Symbol",
            remove_non_ensembl_genes: bool = False,
            custom_order: bool = False,
            output_figure_dir: str = ".",
            output_data_dir: str = "."
    ):
        self.study_ids: list[str] = study_ids if isinstance(study_ids, list) else [study_ids]
        self.cbioportal_data_out_dir = cbioportal_data_out_dir
        self.merge_on_gene_name_type = merge_on_gene_name_type
        self.merge_on_gene_name_type_upset = merge_on_gene_name_type_upset
        self.remove_non_ensembl_genes = remove_non_ensembl_genes
        self.custom_order = custom_order
        self.output_figure_dir = output_figure_dir
        self.output_data_dir = output_data_dir

        assert self.merge_on_gene_name_type in ["Symbol", "Ensembl"], "merge_on_gene_name_type must be either 'Symbol' or 'Ensembl'"
        assert self.merge_on_gene_name_type_upset in ["Symbol", "Ensembl"], "merge_on_gene_name_type_upset must be either 'Symbol' or 'Ensembl'"

        if self.merge_on_gene_name_type_upset == "Symbol":
            self.cosmic_gene_column = "GENE_SYMBOL"
            self.oncokb_gene_column = "Hugo Symbol"
        elif self.merge_on_gene_name_type_upset == "Ensembl":
            self.cosmic_gene_column = "Ensembl_Gene_ID"
            self.oncokb_gene_column = "Ensembl_Gene_ID"

        self.columns_to_keep = ['Tumor_Sample_Barcode', 'Hugo_Symbol', 'Entrez_Gene_Id', 'Consequence']

        # typedef helpers
        _df_dict_2 = defaultdict[str, pd.DataFrame | defaultdict[str, pd.DataFrame]]
        _df_dict_3 = defaultdict[str, pd.DataFrame | defaultdict[str, defaultdict[str, pd.DataFrame]]]

        self.big_combined_df_dict: _df_dict_2 = nested_defaultdict()  # todo: can be local

        self.unexpressed_genes_dict: _df_dict_3 = nested_defaultdict()  # todo: not used at all

        self.merged_before_heatmap_df_dict: _df_dict_3 = nested_defaultdict()  # todo: can be local

        self.heatmap_df_dict: _df_dict_3 = nested_defaultdict()  # todo: can be local

        self.pivot_df_dict: _df_dict_3 = nested_defaultdict()

        self.pivot_df_summary_across_samples_dict: _df_dict_2 = nested_defaultdict()

        # various things defined by functions later

        # noinspection PyTypeChecker
        self.top_mutant_gene_list: list[str] = None
        self.top_mutant_gene_list_ensembl_to_hugo: dict[str, str] | None = None
        self.df_collection: dict[str, dict[str, pd.DataFrame]] = {}
        # noinspection PyTypeChecker
        self.big_combined_df: pd.DataFrame = None
        # noinspection PyTypeChecker
        self.column_for_merging: str = None
        self.upset_data2: pd.DataFrame | None = None
        self.study_id_of_interest: str = study_ids[0] if study_ids else None
        # noinspection PyTypeChecker
        self.gene_list_dict: dict[str, list[str]] = None

        # various things set externally
        self.cosmic_gene_census_df: pd.DataFrame | None = None
        self.cosmic_hallmark_df: pd.DataFrame | None = None
        self.oncokb_df: pd.DataFrame | None = None

    def download_cbioportal_data(self, verbose: bool = False):
        # Check if git lfs is installed
        try:
            result = subprocess.run(['git', 'lfs', 'version'], capture_output=True, text=True, check=True)
            git_lfs_installed = True
        except subprocess.CalledProcessError as e:
            print("Git LFS is not installed or not found in PATH. Proceeding to install without it")
            git_lfs_installed = False

        for study_id in self.study_ids:
            if git_lfs_installed and False: # TODO: maybe fix this, it is currently broken and downloads EVERYTHING
                # Install with git lfs
                datahub_path = ""  # "" if not cloned
                if not os.path.exists(datahub_path):
                    subprocess.run(["git", "lfs", "install", "--skip-repo", "--skip-smudge"], capture_output=True, text=True)  # !git lfs install --skip-repo --skip-smudge
                    result = subprocess.run(["git", "clone", "https://github.com/cBioPortal/datahub.git"], capture_output=True, text=True)  # !git clone https://github.com/cBioPortal/datahub.git
                    datahub_path = "datahub"

                    if result.returncode == 0:
                        print("Datahub repository cloned")
                        print(result.stdout)
                    else:
                        print(f"Cloning datahub repository failed")
                        print(result.stderr)

                os.chdir(datahub_path)
                subprocess.run(["git", "lfs", "install", "--local", "--skip-smudge"], capture_output=True, text=True)  # !git lfs install --local --skip-smudge
                result = subprocess.run(["git", "lfs", "pull", f"public/{study_id}"], capture_output=True, text=True)  # !git lfs pull -I public/{study_id}

                if result.returncode == 0:
                    print(f"All downloading done for {study_id}")
                    print(result.stdout)
                else:
                    print(f"Downloading {study_id} failed")
                    print(result.stderr)

            else:
                # Install without git lfs
                file_types = ["mutations", "cna", "sv", "clinical_patient", "clinical_sample"]  # mutations; sv; cna; clinical_patient; clinical_sample

                os.makedirs(self.cbioportal_data_out_dir, exist_ok=True)

                for file_type in file_types:
                    try:
                        data_folder = os.path.join(self.cbioportal_data_out_dir, study_id)
                        filename = os.path.join(data_folder, f'{file_type}.txt')

                        if os.path.exists(filename):
                            print(f"File {filename} already exists")
                            continue

                        # URL of the file to be downloaded
                        url = f"https://raw.githubusercontent.com/cBioPortal/datahub/master/public/{study_id}/data_{file_type}.txt"
                        # https://raw.githubusercontent.com/cBioPortal/datahub/master/public/msk_impact_2017/data_clinical_patient.txt

                        # Send a GET request to the URL
                        response = requests.get(url)

                        lines = response.content.decode().splitlines(keepends=True)

                        # Extract relevant information from the file
                        version = lines[0].strip().split(' ')[1]
                        oid = lines[1].strip().split(':')[1].strip()
                        size = int(lines[2].strip().split(' ')[1])

                        # Create the JSON object
                        lfs_metadata = {
                            "operation": "download",
                            "transfer": ["basic"],
                            "objects": [
                                {"oid": oid, "size": size}
                            ]
                        }

                        # Convert the dictionary to a JSON string
                        lfs_metadata_json = json.dumps(lfs_metadata)

                        github_url = f"https://github.com/cBioPortal/datahub.git/info/lfs/objects/batch"

                        curl_command = [
                            "curl",
                            "-X", "POST",
                            "-H", "Accept: application/vnd.git-lfs+json",
                            "-H", "Content-type: application/json",
                            "-d", lfs_metadata_json,
                            github_url
                        ]

                        result = subprocess.run(curl_command, capture_output=True, text=True)
                        response_json = json.loads(result.stdout)

                        href = response_json["objects"][0]["actions"]["download"]["href"]

                        response = requests.get(href)

                        if not os.path.exists(data_folder):
                            os.makedirs(data_folder)

                        # Save the file content
                        with open(filename, 'wb') as file:
                            file.write(response.content)

                        if verbose:
                            print(f"File downloaded and saved as {filename}")

                    except Exception as e:
                        print(f"Error downloading {file_type} data: {e}")

                if verbose:
                    print(f"All downloading done for {study_id}, {file_type}")

            if verbose:
                print(f"All downloading done for {study_id}")

        if verbose:
            print("All downloading done")

    def pick_top_n_most_mutated_genes(self, adata: ad.AnnData, n: int = 10):
        # # Pick the top 10 mutated genes (aggregating mutations per gene)
        if self.merge_on_gene_name_type == "Ensembl":
            n_ = adata.uns['combined_genes'].iloc[:n]
            self.top_mutant_gene_list: list[str] = n_.apply(get_valid_ensembl_gene_id_bulk(n_), axis=1).tolist()
            if self.remove_non_ensembl_genes:
                self.top_mutant_gene_list: list[str] = [gene for gene in self.top_mutant_gene_list if "ENSG" not in gene]
            self.top_mutant_gene_list_ensembl_to_hugo: dict[str, str] = dict(zip(self.top_mutant_gene_list, list(adata.uns['combined_genes']['gene_name'][:n])))
        elif self.merge_on_gene_name_type == "Symbol":
            self.top_mutant_gene_list: list[str] = adata.uns['combined_genes']['gene_name'].iloc[:n].tolist()
            self.top_mutant_gene_list_ensembl_to_hugo = None

    def _create_single_study_dataframe(self, study_id: str) -> pd.DataFrame:
        global cancer_type_to_tissue_dictionary

        data_folder = os.path.join(self.cbioportal_data_out_dir, study_id)

#!!!
        with open(os.path.join(data_folder, "mutations.txt"), 'r') as file:
            first_line = file.readline()
            if first_line.startswith('#'):
                # If the first line starts with '#', filter out all lines starting with '#'
                filtered_lines = [line for line in file if not line.startswith('#')]
                
                # Step 2: Read the filtered lines into a DataFrame
                from io import StringIO
                filtered_content = StringIO(''.join(filtered_lines))
                mutation_df = pd.read_csv(filtered_content, sep="\t")
            else:
                # If the first line does not start with '#', read the file normally
                mutation_df = pd.read_csv(os.path.join(data_folder, "mutations.txt"), sep="\t")
#!!!
        
        sample_df = pd.read_csv(os.path.join(data_folder, "clinical_sample.txt"), sep="\t")

        self.df_collection[study_id]['mutations'] = mutation_df
        self.df_collection[study_id]['samples'] = sample_df

        if self.merge_on_gene_name_type == "Ensembl":
            if 'Gene' in mutation_df.columns and not mutation_df['Gene'].isna().all() and mutation_df['Gene'].str.startswith('ENSG').any():
                mutation_df.rename(columns={'Gene': 'Ensembl_Gene_ID'}, inplace=True)
                if self.remove_non_ensembl_genes:
                    mutation_df = mutation_df[mutation_df['Ensembl_Gene_ID'].str.startswith('ENSG')]
            elif 'Transcript_ID' in mutation_df.columns and mutation_df['Transcript_ID'].str.startswith('ENST').any():
                print("Fetching gene IDs from Ensembl")
                mutation_df["Ensembl_Gene_ID"] = mutation_df.progress_apply(
                    get_valid_ensembl_gene_id_bulk(mutation_df),
                    axis=1,
                    transcript_column="Transcript_ID",
                    gene_column='Hugo_Symbol'
                )
                if self.remove_non_ensembl_genes:
                    mutation_df = mutation_df[mutation_df['Ensembl_Gene_ID'].str.startswith('ENSG')]
            else:
                self.merge_on_gene_name_type = "Symbol"
                print("No Ensembl gene IDs found in the mutation data. Merging on gene symbol instead.")

        if self.merge_on_gene_name_type == "Ensembl":
            self.column_for_merging = "Ensembl_Gene_ID"
            self.columns_to_keep.append(self.column_for_merging)
        elif self.merge_on_gene_name_type == "Symbol":
            self.column_for_merging = "Hugo_Symbol"

#!!!
        def join_unique_string_values(series):
            if series.isnull().all():
                return np.nan
            else:
                return ','.join(series.dropna().unique())

        if self.merge_on_gene_name_type == "Ensembl":
            aggregated_df = mutation_df.groupby(['Tumor_Sample_Barcode', self.column_for_merging]).agg({
                'Hugo_Symbol': lambda x: ','.join(x.unique()),
                'Entrez_Gene_Id': lambda x: ','.join(map(str, x.unique())),
                'Consequence': join_unique_string_values
            }).reset_index()
        elif self.merge_on_gene_name_type == "Symbol":
            aggregated_df = mutation_df.groupby(['Tumor_Sample_Barcode', self.column_for_merging]).agg({
                'Entrez_Gene_Id': lambda x: ','.join(map(str, x.unique())),
                'Consequence': join_unique_string_values
            }).reset_index()
        else:
            raise AssertionError(f"Invalid merge_on_gene_name_type: {self.merge_on_gene_name_type}")

        aggregated_df['Consequence'] = aggregated_df['Consequence'].apply(
            lambda x: "Multiple_consequences" if isinstance(x, str) and ',' in x else x
        )

        occurrences_df = aggregated_df.groupby([self.column_for_merging, 'Tumor_Sample_Barcode']).size().reset_index(name='mutation_occurrences')

        final_df = pd.merge(aggregated_df, occurrences_df, on=[self.column_for_merging, 'Tumor_Sample_Barcode'])

        # add CNA and SV info
        if os.path.exists(os.path.join(data_folder, "cna.txt")):
            cna_df = pd.read_csv(os.path.join(data_folder, "cna.txt"), sep="\t")
            self.df_collection[study_id]['cna'] = cna_df

            # Exclude 'Hugo_Symbol' column
            columns_to_transform = self.df_collection[study_id]['cna'].columns.difference(['Hugo_Symbol'])

            # Apply binary transformation to the selected columns
            df_binary = self.df_collection[study_id]['cna'][columns_to_transform].map(lambda x: 1 if (pd.notna(x) and x != 0) else x)  # # df_binary = self.df_collection[study_id]['cna'][columns_to_transform].apply(lambda col: col.map(lambda x: 1 if (pd.notna(x) and x != 0) else x))

            # Add 'Hugo_Symbol' column back to the DataFrame
            df_binary.insert(0, 'Hugo_Symbol', self.df_collection[study_id]['cna']['Hugo_Symbol'])

            # Reassign the transformed DataFrame to the collection
            self.df_collection[study_id]['cna_binary'] = df_binary

            melted_cna = self.df_collection[study_id]['cna_binary'].melt(id_vars=['Hugo_Symbol'], var_name='Tumor_Sample_Barcode', value_name='cna_occurrences')

            final_df = pd.merge(final_df, melted_cna[['Hugo_Symbol', 'Tumor_Sample_Barcode', 'cna_occurrences']], on=['Hugo_Symbol', 'Tumor_Sample_Barcode'], how='outer')

        if os.path.exists(os.path.join(data_folder, "sv.txt")):
            sv_df = pd.read_csv(os.path.join(data_folder, "sv.txt"), sep="\t")
            self.df_collection[study_id]['sv'] = sv_df

            # Melt the DataFrame to combine Site1_Hugo_Symbol and Site2_Hugo_Symbol into a single column
            melted_sv = pd.melt(self.df_collection[study_id]['sv'], id_vars=['Sample_Id'], value_vars=['Site1_Hugo_Symbol', 'Site2_Hugo_Symbol'],
                                var_name='site', value_name='Hugo_Symbol')

            # Drop duplicate rows to ensure each Hugo_Symbol is only counted once per Sample_Id
            melted_sv = melted_sv.drop_duplicates(subset=['Sample_Id', 'Hugo_Symbol'])

            # Count the occurrences of each Hugo_Symbol in each Sample_Id
            sv_occurrences = melted_sv.groupby(['Hugo_Symbol', 'Sample_Id']).size().reset_index(name='sv_occurrences')

            # Rename columns to match the desired output
            sv_occurrences = sv_occurrences.rename(columns={'Sample_Id': 'Tumor_Sample_Barcode'})

            final_df = pd.merge(final_df, sv_occurrences[['Hugo_Symbol', 'Tumor_Sample_Barcode', 'sv_occurrences']], on=['Hugo_Symbol', 'Tumor_Sample_Barcode'], how='outer')

        final_df['study_id'] = study_id

        if 'Sample Identifier' in sample_df.columns:
            sample_identifier_column = 'Sample Identifier'
        elif '#Sample Identifier' in sample_df.columns:
            sample_identifier_column = '#Sample Identifier'
        else:
            raise AssertionError("Sample Identifier column not found in the sample dataframe")

#!!!
        if 'Cancer Type' not in sample_df.columns:
            sample_df['Cancer Type'] = np.nan
        if 'Cancer Type Detailed' not in sample_df.columns:
            sample_df['Cancer Type Detailed'] = np.nan
#!!!
        final_df = pd.merge(final_df, sample_df[[sample_identifier_column, 'Cancer Type', 'Cancer Type Detailed']], 
                            left_on='Tumor_Sample_Barcode', right_on=sample_identifier_column, how='left')

        final_df.rename(columns={'Cancer Type': 'cancer_type', 'Cancer Type Detailed': 'cancer_type_detailed'}, inplace=True)

        final_df['tissue'] = final_df['cancer_type'].map(cancer_type_to_tissue_dictionary).fillna('unclassified')

        # Drop the redundant SAMPLE_ID column
        final_df.drop(columns=[sample_identifier_column], inplace=True)

        return final_df

    def create_study_dataframes(self):
        dataframes = []

        self.df_collection: dict[str, dict[str, pd.DataFrame]] = {}

        for study_id in self.study_ids:
            self.df_collection[study_id] = {}
            if study_id == "msk_impact_2017":
                with open(f"{self.cbioportal_data_out_dir}/{study_id}/mutations.txt", 'r') as file:
                    lines = file.readlines()[1:]

                if lines[0].split('\t')[0] == "Hugo_Symbol":
                    # Write the remaining lines back to the file
                    with open(f"{self.cbioportal_data_out_dir}/{study_id}/mutations.txt", 'w') as file:
                        file.writelines(lines)

            final_df = self._create_single_study_dataframe(study_id=study_id)
            dataframes.append(final_df)

        self.big_combined_df = pd.concat(dataframes, ignore_index=True)

    def plot_heatmap(
            self,
            stratification: str = "tissue",
            selected_group: str | None = None,
            selected_group_category: str | None = None,
            variation_type: Literal[
                "mutation_occurrences",
                "cna_nonbinary",
                "sv_occurrences",
                "cna_occurrences",
                "Consequence"
            ] = "mutation_occurrences",
            # todo this is only used in one place to check if it is "Hugo_Symbol", should it be used more widely or should it be removed?
            display_name: str = "Hugo_Symbol",
            df: pd.DataFrame | None = None
    ):
        if variation_type == "cna_nonbinary" or variation_type == "Consequence":
            assert stratification == "sample", "Stratification must be 'sample' for cna_nonbinary and Consequence variations"
        if variation_type != "cna_nonbinary":
            simple_merge_by_stratification: dict[str, list[str]] = {
                "tissue":               ["Tumor_Sample_Barcode", "study_id", "cancer_type", "tissue"],
                "cancer_type":          ["Tumor_Sample_Barcode", "study_id", "cancer_type"],
                "cancer_type_detailed": ["Tumor_Sample_Barcode", "study_id", "cancer_type_detailed"],
                "study_id":             ["Tumor_Sample_Barcode", "study_id"],
                "sample":               ["Tumor_Sample_Barcode"]
            }
            # stratify by merging specific columns
            if stratification in simple_merge_by_stratification:
                merge_on: list[str] = simple_merge_by_stratification[stratification]

                if selected_group_category is None:  #* no filtering
                    final_df = self.big_combined_df
                else:
                    self.big_combined_df_dict[selected_group_category][selected_group] = self.big_combined_df[self.big_combined_df[selected_group_category] == selected_group]
                    final_df = self.big_combined_df_dict[selected_group_category][selected_group]

                columns_to_keep_copy = self.columns_to_keep.copy()

                unique_samples_info = final_df[['Tumor_Sample_Barcode', 'cancer_type', 'cancer_type_detailed', 'study_id', 'tissue']].drop_duplicates()

                hugo_mask = final_df['Hugo_Symbol'].isin([gene for gene in self.top_mutant_gene_list if not gene.startswith('ENSG')])

                if self.merge_on_gene_name_type == "Ensembl":
                    ensg_mask = final_df['Ensembl_Gene_ID'].isin([gene for gene in self.top_mutant_gene_list if gene.startswith('ENSG')])
                    combined_mask = ensg_mask | hugo_mask
                elif self.merge_on_gene_name_type == "Symbol":
                    combined_mask = hugo_mask
                else:
                    raise AssertionError(f"Invalid merge_on_gene_name_type: {self.merge_on_gene_name_type}")

                filtered_genes_df: pd.DataFrame = final_df[combined_mask]

                if self.merge_on_gene_name_type == "Ensembl":
                    existing_genes = set(filtered_genes_df['Ensembl_Gene_ID']).union(set(filtered_genes_df['Hugo_Symbol']))
                elif self.merge_on_gene_name_type == "Symbol":
                    existing_genes = set(filtered_genes_df['Hugo_Symbol'])
                else:
                    raise AssertionError(f"Invalid merge_on_gene_name_type: {self.merge_on_gene_name_type}")

                unexpressed_genes = [gene for gene in self.top_mutant_gene_list if gene not in existing_genes]

                # Get all unique Tumor_Sample_Barcode from the original DataFrame
                all_samples = final_df[merge_on].drop_duplicates()

                all_samples = pd.merge(all_samples, unique_samples_info, on=merge_on, how='left')

                if variation_type not in columns_to_keep_copy:
                    columns_to_keep_copy.append(variation_type)

                must_keep = ["study_id", "tissue", "cancer_type", "cancer_type_detailed"]
                for column_name in must_keep:
                    if column_name in merge_on and column_name not in columns_to_keep_copy:
                        columns_to_keep_copy.append(column_name)

                # Merge the filtered genes DataFrame with all samples to ensure all samples are included
                merged_df = pd.merge(all_samples, filtered_genes_df[columns_to_keep_copy], on=merge_on, how='left', suffixes=('_sample', '_gene'))

                if selected_group_category is None:  #* no filtering
                    self.merged_before_heatmap_df_dict[variation_type] = merged_df
                    self.unexpressed_genes_dict[variation_type] = unexpressed_genes
                    self.heatmap_df_dict[variation_type] = self.merged_before_heatmap_df_dict[variation_type].groupby([self.column_for_merging, stratification])[variation_type].sum().reset_index()
                    df_for_heatmap_very_final: pd.DataFrame = self.heatmap_df_dict[variation_type]
                else:
                    self.merged_before_heatmap_df_dict[selected_group_category][selected_group][variation_type] = merged_df
                    self.unexpressed_genes_dict[selected_group_category][selected_group][variation_type] = unexpressed_genes
                    if stratification == "sample":
                        self.heatmap_df_dict[selected_group_category][selected_group][variation_type] = self.merged_before_heatmap_df_dict[selected_group_category][selected_group][variation_type]
                    else:
                        self.heatmap_df_dict[selected_group_category][selected_group][variation_type] = self.merged_before_heatmap_df_dict[selected_group_category][selected_group][variation_type].groupby([self.column_for_merging, stratification])[variation_type].sum().reset_index()
                    df_for_heatmap_very_final: pd.DataFrame = self.heatmap_df_dict[selected_group_category][selected_group][variation_type]

                df = df_for_heatmap_very_final.copy()

            #* plot heatmap code
            pivot_column = stratification

            # custom stratification
            if stratification == "gene_database":
                self.upset_data2, _ = self.make_upset_df(**self.gene_list_dict)
                df = self.upset_data2
                # DO NOT CHANGE TO `is True` - it will not work. For some reason, some deserialization
                # code somewhere makes MULTIPLE INSTANCES of True, so `is` fails to operate
                df = df[df['RNAseq'] == True]

                # Drop the RNAseq column
                df = df.drop(columns=['RNAseq']).astype(int)

                pivot_df1 = df

                title = f'Heatmap of Mutant genes across {stratification}'
            elif "cosmic_gene_census" in stratification or "hallmark" in stratification:
                df = self.pivot_df_summary_across_samples_dict[self.study_id_of_interest]["mutation_occurrences"]
                unique_tumor_types = set()
                for types in df[stratification].dropna():
                    unique_tumor_types.update(types.split(','))

                unique_tumor_types.discard('nan')

                unique_tumor_types = sorted(unique_tumor_types)

                # Create a new DataFrame for the heatmap
                heatmap_data = pd.DataFrame(0, index=df.index, columns=unique_tumor_types)

                for gene, types in df[stratification].dropna().items():
                    for t in types.split(','):
                        if t:
                            heatmap_data.at[gene, t] = 1

                if 'nan' in heatmap_data.columns:
                    heatmap_data = heatmap_data.drop(columns=['nan'])

                pivot_df1 = heatmap_data
                title = f'Heatmap of Mutant genes across {stratification}'
            # simple stratification (round 2)
            else:
                if stratification in {'tissue', 'cancer_type', 'cancer_type_detailed', 'study_id'}:
                    pass
                elif stratification == 'sample':
                    pivot_column = 'Tumor_Sample_Barcode'
                else:
                    raise ValueError("Invalid stratification value. Please choose from 'tissue', 'cancer_type', 'cancer_type_detailed', 'study_id', 'sample'")

                pivot_df1 = df.pivot(index=self.column_for_merging, columns=pivot_column, values=variation_type)

                pivot_df1 = pivot_df1.dropna(how='all')

                sorted_columns = pivot_df1.count().sort_values(ascending=False).index
                pivot_df1 = pivot_df1[sorted_columns]

                # unexpressed_genes IS in fact always bound by the time we get here
                # noinspection PyUnboundLocalVariable
                if unexpressed_genes:
                    # noinspection PyUnboundLocalVariable
                    new_rows = pd.DataFrame(np.nan, index=unexpressed_genes, columns=pivot_df1.columns)
                    pivot_df1 = pd.concat([pivot_df1, new_rows])
                title = f'Heatmap of Gene mutations per gene across {stratification}'

                if selected_group:
                    title = title + ' - ' + selected_group

            if self.column_for_merging == "Ensembl_Gene_ID" and display_name == "Hugo_Symbol":
                if self.top_mutant_gene_list_ensembl_to_hugo is None:
                    print("Fetching gene names from Ensembl...")
                    map_ = get_ensembl_gene_name_bulk(pivot_df1.index.tolist())
                    print("Fetched gene names from Ensembl")
                    pivot_df1.rename(index=map_, inplace=True)
                else:
                    pivot_df1.rename(index=self.top_mutant_gene_list_ensembl_to_hugo, inplace=True)

        else:  # variation_type == "cna_nonbinary"
            assert stratification == "sample", "stratification must be 'sample' for CNA data"
            assert selected_group_category == "study_id", "selected_group_category must be 'study_id' for CNA data"
            pivot_df1 = self.df_collection[selected_group]['cna'].copy()
            pivot_df1.set_index('Hugo_Symbol', inplace=True)
            pivot_df1 = pivot_df1[pivot_df1.index.isin(self.top_mutant_gene_list)]

            pivot_df1 = pivot_df1.reset_index()
            if 'Hugo_Symbol' not in pivot_df1.columns:
                pivot_df1['Hugo_Symbol'] = pivot_df1.index

            # Iterate over the top_mutant_gene_list and add missing entries
            for gene in self.top_mutant_gene_list:
                if gene not in pivot_df1['Hugo_Symbol'].values:
                    new_row = pd.Series({col: np.nan for col in pivot_df1.columns}, name=gene)
                    new_row['Hugo_Symbol'] = gene
                    pivot_df1 = pivot_df1.append(new_row)

            # Set 'Hugo_Symbol' back as index if needed
            pivot_df1 = pivot_df1.set_index('Hugo_Symbol')

            title = "Heatmap of CNA data across samples"

        if pivot_df1.isna().all().all():
            print(f"No data to plot for {stratification}")
            return None

        nas_present = True

        # ensure pivot_df1 is sorted by columns before plotting
        pivot_df1: pd.DataFrame
        pivot_df1.sort_index(axis="columns", inplace=True)

        # limit to first 500 columns
        render_divider_lines = True
        render_column_ids = pivot_df1.shape[1] < 100
        if pivot_df1.shape[1] > 372:  # 372 is fine, 373 is not. There's something wrong with pyplot...
            print("Warning: Too many columns to plot. Limiting to first 372 columns")
            pivot_df1 = pivot_df1.iloc[:, :372]
            render_divider_lines = False

        if variation_type == "cna_nonbinary":
            min_value = -3
            max_value = 2

            levels = list(range(min_value + 1, max_value + 1))
            pivot_df1 = pivot_df1.fillna(min_value)

            colors_list = plt.get_cmap('RdBu_r', max_value - min_value + 1)(range(max_value - min_value + 1))
            colors_list = np.vstack(([[0.5, 0.5, 0.5, 0.3]], colors_list[1:]))  # Grey color for -3
            cmap = ListedColormap(colors_list)

            # Define the norm with the diverging palette centered at 0
            norm = TwoSlopeNorm(vmin=min_value, vcenter=0, vmax=max_value)

        elif variation_type == "Consequence":
            consequences: list[str|float] = list(self.big_combined_df['Consequence'].unique())

            colors_list = plt.get_cmap('tab20', len(consequences))(range(len(consequences)))

            # if consequences contains nan, ensure the nan value is at the beginning
            if np.nan in consequences:
                colors_list = np.vstack(([[1.0, 1.0, 1.0, 0.3]], colors_list[:-1]))
                consequences = [np.nan] + sorted(v for v in consequences if not isinstance(v, float))
            else:
                consequences.sort()
                nas_present = False

            cmap = ListedColormap(colors_list)
            min_value = 0
            max_value = len(consequences)
            norm = BoundaryNorm(boundaries=np.arange(min_value - 0.5, max_value + 0.5, 1), ncolors=cmap.N, clip=False)
            levels = list(range(min_value, max_value))

            string_to_int: dict[str | float, int] = {consequence: i for i, consequence in enumerate(consequences)}

            pivot_df1 = pivot_df1.map(lambda x: string_to_int[x])

        else:
            min_value = int(pivot_df1.min().min())

            if pivot_df1.isna().sum().sum() != 0:
                min_value -= 1
            else:
                nas_present = False

            max_value = max(int(pivot_df1.max().max()), 1)

            levels = list(range(min_value, max_value + 1))
            pivot_df1 = pivot_df1.fillna(min_value)

            # Create a custom colormap
            colors_list = plt.get_cmap('Reds', len(levels))(range(len(levels)))
            if nas_present:
                colors_list = np.vstack(([[0.5, 0.5, 0.5, 0.3]], colors_list))  # Grey color for -1
            cmap = ListedColormap(colors_list)

            # Define the norm, with vmin set to -1 and vmax to max_value
            norm = BoundaryNorm(boundaries=np.arange(min_value - 0.5, max_value + 1.5, 1), ncolors=cmap.N, clip=False)

        # Create a figure
        plt.figure(figsize=(18, 4))

        # Display the heatmap with the filled array
        plt.imshow(pivot_df1, cmap=cmap, norm=norm, aspect='auto')

        colorbar_label = 'Number of Mutations'
        if variation_type == "Consequence":
            colorbar_label = 'Consequence Type'
        else:
            # Add colorbar with ticks for all levels, including the placeholder for -1 (NaN)
            levels = ints_between(min_value, max_value, 25, 7)

        cbar = plt.colorbar(label=colorbar_label, ticks=levels)

        labels: list[str | int] = levels.copy()
        if nas_present:
            labels[0] = 'NaN'

        if variation_type == "Consequence":
            # Guaranteed to be bound by code flow
            # noinspection PyUnboundLocalVariable
            # start at 0 if there are no NaN values, otherwise at 1
            for i, consequence in enumerate(consequences[int(nas_present):]):
                labels[i + int(nas_present)] = consequence

        cbar.ax.set_yticklabels(labels)

        plt.grid(which='both', axis='both' if render_divider_lines else 'y', color='black', linestyle='-', linewidth=0.5)
        if render_column_ids:
            x_labels = pivot_df1.columns
        else:
            x_labels = [""] * len(pivot_df1.columns)
        plt.xticks(np.arange(len(pivot_df1.columns)), x_labels, rotation=90)
        plt.yticks(np.arange(len(pivot_df1.index)), pivot_df1.index)

        plt.gca().set_xticks(np.arange(-0.5, len(pivot_df1.columns), 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, len(pivot_df1.index), 1), minor=True)

        plt.grid(which='major', color='white', linestyle='-', linewidth=0.5, alpha = 0)
        plt.gca().tick_params(which='minor', size=0)

        plt.xlabel(stratification)
        plt.ylabel('Genes')
        plt.title(title)

        filename = f'Heatmap_{stratification}'
        if selected_group:
            filename = f'{filename}_{selected_group}'

        filepath = os.path.join(self.output_figure_dir, f'{filename}.png')

        plt.savefig(filepath, bbox_inches='tight')

        plt.show()

        if nas_present:
            pivot_df1 = pivot_df1.replace(min_value, np.nan)

        if (stratification == "gene_database") or ("cosmic_gene_census" in stratification) or ("hallmark" in stratification):
            self.pivot_df_dict[stratification] = pivot_df1
        else:
            if selected_group_category is None:  #* no filtering
                self.pivot_df_dict[variation_type] = pivot_df1
            else:
                self.pivot_df_dict[selected_group_category][selected_group][variation_type] = pivot_df1

    def create_summary_table_across_samples(self, study_id: str, variation_type: str = "mutation_occurrences"):
        pivot_df1 = self.pivot_df_dict['study_id'][study_id][variation_type]
        total_mutations = pivot_df1.sum(axis=1)

        # Calculate the number of samples with at least one mutation for each gene
        samples_with_mutations = (pivot_df1.notna() & (pivot_df1 > 0)).sum(axis=1)

        fraction_of_samples_with_mutations = (samples_with_mutations / pivot_df1.shape[1]).round(2)

        # Create a new DataFrame with the results
        pivot_df1_summary = pd.DataFrame({
            'Total_Mutations': total_mutations,
            'Samples_With_At_Least_1_Mutation': samples_with_mutations,
            'Fraction_Of_Samples_With_At_Least_1_Mutation': fraction_of_samples_with_mutations
        })

        self.pivot_df_summary_across_samples_dict[study_id][variation_type] = pivot_df1_summary

    def plot_fraction_of_samples_over_genes_bargraph(self, study_id: str, variation_type: str = "mutation_occurrences"):
        if variation_type == "combined":
            self.big_combined_df['mutation_occurrences_binary'] = self.big_combined_df['mutation_occurrences'].apply(lambda x: 1 if (pd.notna(x) and x != 0) else 0)
            self.big_combined_df['cna_occurrences_binary'] = self.big_combined_df['cna_occurrences'].apply(lambda x: 1 if (pd.notna(x) and x != 0) else 0)
            self.big_combined_df['sv_occurrences_binary'] = self.big_combined_df['sv_occurrences'].apply(lambda x: 1 if (pd.notna(x) and x != 0) else 0)

            # Group by Hugo_Symbol and sum the binary columns
            summary_df = self.big_combined_df.groupby('Hugo_Symbol').agg({
                'mutation_occurrences_binary': 'sum',
                'cna_occurrences_binary': 'sum',
                'sv_occurrences_binary': 'sum'
            }).reset_index()

            # Calculate the intersection counts
            intersect_df = self.big_combined_df.groupby('Hugo_Symbol').apply(
                lambda x: pd.Series({
                    'mutation_and_cna': ((x['mutation_occurrences_binary'] == 1) & (x['cna_occurrences_binary'] == 1)).sum(),
                    'mutation_and_sv': ((x['mutation_occurrences_binary'] == 1) & (x['sv_occurrences_binary'] == 1)).sum(),
                    'cna_and_sv': ((x['cna_occurrences_binary'] == 1) & (x['sv_occurrences_binary'] == 1)).sum(),
                    'mutation_and_cna_and_sv': ((x['mutation_occurrences_binary'] == 1) & (x['cna_occurrences_binary'] == 1) & (x['sv_occurrences_binary'] == 1)).sum()
                })
            ).reset_index()

            # Merge the summaries with the intersection counts
            df = pd.merge(summary_df, intersect_df, on='Hugo_Symbol')

            df = df.loc[df['Hugo_Symbol'].isin(self.top_mutant_gene_list)]

            num_samples = len(self.big_combined_df['Tumor_Sample_Barcode'].unique())

            df.update(df.iloc[:, 1:].div(num_samples))

            fig, ax = plt.subplots(figsize=(10, 6))

            # Define colors for each category
            colors = {
                'mutation_occurrences_binary': 'red',
                'cna_occurrences_binary': 'yellow',
                'sv_occurrences_binary': 'blue',
                'mutation_and_cna': 'orange',
                'mutation_and_sv': 'purple',
                'cna_and_sv': 'green',
                'mutation_and_cna_and_sv': 'brown'
            }

            # Initial bottom position for stacking
            bottom = np.zeros(len(df))

            # Plot each category as a stacked bar
            for column, color in colors.items():
                ax.bar(df['Hugo_Symbol'], df[column], label=column, color=color, bottom=bottom)
                bottom += df[column]  # Update the bottom position for the next category

            # Add legend and labels
            ax.legend()
            ax.set_xlabel('Hugo_Symbol')
            ax.set_ylabel('Occurrences')
            ax.set_title('Genetic Occurrences and Their Intersections')
        else:
            df = self.pivot_df_summary_across_samples_dict[study_id][variation_type]

            plt.figure(figsize=(12, 6))
            df['Fraction_Of_Samples_With_At_Least_1_Mutation'].plot(kind='bar', color='skyblue')

            plt.xlabel('Gene')
            plt.ylabel('Detection Fraction')
            plt.title('Detection Fraction of Genes in Samples')

        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.output_figure_dir, f'Mutation_rate_across_samples_bar_graph.png'))
        plt.show()

    @staticmethod
    def extract_first_ensg(synonyms: str) -> str:
        # Split the string by commas
        synonyms_list = synonyms.split(',')
        # Find the first entry that starts with 'ENSG'
        for synonym in synonyms_list:
            if synonym.startswith('ENSG'):
                return synonym
        return "Unknown"  # Return None if no 'ENSG' entry is found

    @staticmethod
    def make_upset_df(**gene_list_dict: list[str]) -> tuple[pd.DataFrame, dict[str, set[str]]]:
        gene_set_dict = {name: set(gene_list) for name, gene_list in gene_list_dict.items()}

        all_genes = set().union(*gene_set_dict.values())

        all_genes = sorted(all_genes)

        data = {}
        for key, gene_set in gene_set_dict.items():
            data[key] = [True if gene in gene_set else False for gene in all_genes]

        data['Gene'] = all_genes

        df = pd.DataFrame(data).set_index('Gene')

        return df, gene_set_dict

    def merge_gene_database_with_top_mutated_genes(self, custom_order: bool = False, **gene_list_dict: list[str]) -> pd.Series:
        df, gene_set_dict = self.make_upset_df(**gene_list_dict)

        upset_data = df.groupby(list(gene_set_dict.keys())).size()

        if custom_order:  # takes on the same order as gene_list_dict keys
            index = pd.MultiIndex.from_tuples(upset_data.keys(), names=list(gene_set_dict.keys()))
            upset_data = pd.Series(upset_data, index=index)

            # Sort the Series such that entries containing RNAseq appear first
            upset_data = upset_data.sort_index(level=list(gene_set_dict.keys())[0], ascending=False)  # assumes experiment comes first

        return upset_data

    def make_upset_plot(self):
        assert (
                self.cosmic_gene_census_df is not None
                and self.cosmic_hallmark_df is not None
                and self.oncokb_df is not None
        ), "Please load the cosmic_gene_census_df, cosmic_hallmark_df, and oncokb_df DataFrames first."

        if self.merge_on_gene_name_type_upset == "Symbol":
            if self.merge_on_gene_name_type == "Ensembl":
                top_mutant_gene_list_for_upset = [self.top_mutant_gene_list_ensembl_to_hugo.get(x, x) for x in self.top_mutant_gene_list]
            else:
                top_mutant_gene_list_for_upset = self.top_mutant_gene_list
            cosmic_gene_census_gene_list_for_upset = self.cosmic_gene_census_df['GENE_SYMBOL'].tolist()
            cosmic_hallmark_gene_list_for_upset = self.cosmic_hallmark_df['GENE_SYMBOL'].tolist()
            oncokb_gene_list_for_upset = self.oncokb_df['Hugo Symbol'].tolist()

        elif self.merge_on_gene_name_type_upset == "Ensembl":
            assert self.merge_on_gene_name_type == "Ensembl"
            self.cosmic_gene_census_df['Ensembl_Gene_ID'] = self.cosmic_gene_census_df['SYNONYMS'].apply(self.extract_first_ensg)
            self.cosmic_gene_census_df['Ensembl_Gene_ID'] = self.cosmic_gene_census_df['Ensembl_Gene_ID'].str.replace(r'\.\d+', '', regex=True)
            cosmic_gene_census_gene_list = self.cosmic_gene_census_df['Ensembl_Gene_ID'].tolist()

            self.cosmic_hallmark_df = pd.merge(self.cosmic_hallmark_df, self.cosmic_gene_census_df[['GENE_SYMBOL', 'COSMIC_GENE_ID', 'Ensembl_Gene_ID']], on=['GENE_SYMBOL', 'COSMIC_GENE_ID'], how='left')
            cosmic_hallmark_gene_list = self.cosmic_hallmark_df['Ensembl_Gene_ID'].tolist()

            self.oncokb_df['Ensembl_Gene_ID'] = self.oncokb_df.progress_apply(get_valid_ensembl_gene_id_bulk(self.oncokb_df), axis = 1, transcript_column = "GRCh37 Isoform", gene_column = 'Hugo Symbol')
            oncokb_gene_list = self.oncokb_df['Ensembl_Gene_ID'].tolist()

            if self.remove_non_ensembl_genes:
                top_mutant_gene_list_for_upset = [gene for gene in self.top_mutant_gene_list if gene.startswith('ENSG')]
                cosmic_gene_census_gene_list_for_upset = [gene for gene in cosmic_gene_census_gene_list if gene.startswith('ENSG')]
                cosmic_hallmark_gene_list_for_upset = [gene for gene in cosmic_hallmark_gene_list if gene.startswith('ENSG')]
                oncokb_gene_list_for_upset = [gene for gene in oncokb_gene_list if gene.startswith('ENSG')]

            else:
                top_mutant_gene_list_for_upset = self.top_mutant_gene_list
                cosmic_gene_census_gene_list_for_upset = cosmic_gene_census_gene_list
                cosmic_hallmark_gene_list_for_upset = cosmic_hallmark_gene_list
                oncokb_gene_list_for_upset = oncokb_gene_list

        else:
            raise ValueError("Invalid merge_on_gene_name_type_upset. Please choose from 'Symbol', 'Ensembl'")

        self.gene_list_dict = {
            "RNAseq": top_mutant_gene_list_for_upset,
            "Cosmic_gene_census": cosmic_gene_census_gene_list_for_upset,
            "Cosmic_hallmark_genes": cosmic_hallmark_gene_list_for_upset,
            "OncoKB_cancer_genes": oncokb_gene_list_for_upset
        }

        upset_data = self.merge_gene_database_with_top_mutated_genes(**self.gene_list_dict, custom_order=self.custom_order)

        sort_by = "input" if self.custom_order else "degree"
        upset = UpSet(upset_data, show_counts=True, sort_by = sort_by)
        upset.plot()

        plt.savefig(os.path.join(self.output_figure_dir, 'gene_upset_across_databases.png'))

        plt.show()
