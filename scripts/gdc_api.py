import json
import math

import pandas as pd
from gget.utils import graphql_query, json_list_to_df

GDC_API = "https://api.gdc.cancer.gov/v0/graphql"


QUERY = """
query ConsequencesTable (
    $filters: FiltersArgument
) {
    viewer {
        explore {
            ssms {
                hits(filters: $filters) {
                    edges {
                        node {
                            consequence {
                                hits {
                                    total
                                    edges {
                                        node {
                                            transcript {
                                                id
                                                transcript_id
                                                aa_change
                                                annotation {
                                                    hgvsc
                                                    polyphen_impact
                                                    polyphen_score
                                                    sift_score
                                                    sift_impact
                                                    vep_impact
                                                }
                                                gene {
                                                    gene_id
                                                    symbol
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
"""


def gdc_query(ssm_ids: list[str]) -> pd.DataFrame:
    variables = {
        "filters": {
            "op": "in",
            "content": {
                "field": "ssms.ssm_id",
                "value": ssm_ids
            }
        }
    }

    data = graphql_query(GDC_API, QUERY, variables)

    edges1: list[dict[str, ...]] = data['data']['viewer']['explore']['ssms']['hits']['edges']

    actual_data: list[dict[str, ...]] = sum([e['node']['consequence']['hits']['edges'] for e in edges1], []) # flattened list

    actual_data: list[dict[str, ...]] = [d['node']['transcript'] for d in actual_data]

    if __name__ == "__main__":
        print(json.dumps(actual_data[0], indent=2))

    df: pd.DataFrame = json_list_to_df(
        actual_data,
        [
            ("ssm_id", "id"),
            ("aa_change", "aa_change"),
            ("transcript_id", "transcript_id"),

            ("hgvsc", "annotation.hgvsc"),
            ("polyphen_impact", "annotation.polyphen_impact"),
            ("polyphen_score", "annotation.polyphen_score"),
            ("sift_score", "annotation.sift_score"),
            ("sift_impact", "annotation.sift_impact"),
            ("vep_impact", "annotation.vep_impact"),

            ("gene_id", "gene.gene_id"),
            ("gene_symbol", "gene.symbol"),
        ]
    ).applymap(lambda x: x if x is not None else math.nan)

    df["consequence_id"] = df["ssm_id"].map(lambda x: x.split(":")[1])
    df["ssm_id"] = df["ssm_id"].map(lambda x: x.split(":")[0])

    return df


if __name__ == "__main__":
    # df = gdc_query(["84aef48f-31e6-52e4-8e05-7d5b9ab15087", "edd1ae2c-3ca9-52bd-a124-b09ed304fcc2"])

    MASSIVE_LIST = [] # the whole ssm_id column
    CHUNK_SIZE = 1000

    df_combined = pd.DataFrame()

    for i in range(0, len(MASSIVE_LIST), CHUNK_SIZE):
        chunk = MASSIVE_LIST[i:i+CHUNK_SIZE]
        df = gdc_query(chunk)

        df_combined = pd.concat([df_combined, df])  #! untested
