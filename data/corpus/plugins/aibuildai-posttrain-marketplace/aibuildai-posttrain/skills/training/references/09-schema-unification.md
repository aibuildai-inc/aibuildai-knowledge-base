# Schema unification

Read in step 5 before mixing sources that carry different formats.

Many public datasets cannot be mixed as they are, because their formats do not agree. Make them one shape first (canonicalize):

- OpenAI wrapped tool schema and bare function schema;
- make sure `parameters.type="object"`;
- make sure `properties` is there;
- parse the arguments string into a JSON object;
- keep single tool call and multi tool call apart;
- turn code completion into one function body;
- make answer letter and answer text agree;
- make the system/user/assistant roles agree;
- add EOS to every final output.

A function-calling data script can do this:

1. parse string-form tools;
2. turn them all into `{"type":"function","function":...}`;
3. drop samples with no tool call or with many tool calls;
4. check that arguments is a dictionary;
5. render it as one single `<tool_call>` target.

This kind of unification is often worth more than a few thousand extra rows that nobody treated.

The dataset skill carries the catalog of sources; this card keeps only the format work the workflow needs.
