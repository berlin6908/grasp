"""Frozen names shared by Pipeline manifests and offline evaluation."""

RESULT_VERSION = 29
FULL_DAG_TREATMENT = "full_dag"
NO_SIGNATURE_TREATMENT = "without_relation_signatures"
NO_BINDING_JOINS_TREATMENT = "without_binding_joins"
ASSIGNMENT_SCORING = "full_dag_relation_log_yes_probability_sum"
DAG_SEARCH = "role_dag_binding_conditioned_single_relation_beam"
NO_BINDING_JOINS_DAG_SEARCH = (
    "role_plan_binding_conditioned_single_relation_beam_without_intersection_filter"
)
RELATION_SEARCH = "signature_known_executable_then_typed_output_aware_scoring"
NO_SIGNATURE_RELATION_SEARCH = (
    "all_adjacent_executable_then_untyped_output_aware_scoring"
)
TERMINAL_OUTPUT_POLICY = "ranked_complete_assignments_with_factorized_grounded_programs"
FINAL_JUDGE_MODEL = "gpt-oss:120b"
FINAL_JUDGE_REVISION = "a951a23b46a1f6093dafee2ea481d634b4e31ac720a8a16f3f91e04f5a40ecd9"
FINAL_RESULT_VERSION = 13
