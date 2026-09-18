# Contributing

Keep reusable code under `src/brain2text`, command-line data preparation under
`scripts`, and generated outputs out of version control. Before opening a change:

```bash
python -m compileall src scripts
ruff check src scripts
```

Changes to model architectures or decoding behavior should include a small
reproducible example or a focused test when practical.

The project does not require KenLM. The old KenLM/`pyctcdecode` utilities are kept
only as legacy research references and are not part of the primary LLM pipeline.