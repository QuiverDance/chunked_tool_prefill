# Tool-Blocked Prefill

This project studies and implements prompt prefill work that can overlap tool execution without changing the model's eventual input.

## Language

**Candidate Tool Prefill**:
Verified speculative prefill of historical tool-output candidates while the current tool is running. Only the exact prefix verified against the actual output is reusable.
_Avoid_: BranchFill, predicted output injection, speculative tool result

**History trunk**:
The prompt prefix shared by every candidate branch before predicted tool-output content begins.
_Avoid_: Base prompt, common history

**Speculative trunk**:
An output prefix shared by a group of candidate branches beyond the history trunk.
_Avoid_: Common candidate text, shared guess

**Candidate branch**:
A speculative prompt extension built from one retrieved historical tool output.
_Avoid_: Prediction, generated output

**Branch delta**:
The candidate-specific suffix after its nearest speculative trunk.
_Avoid_: Candidate tail, unique output

**Verified prefix**:
The longest candidate prefix that exactly matches the actual tool-output tokens observed so far.
_Avoid_: Accepted prediction, probable prefix
