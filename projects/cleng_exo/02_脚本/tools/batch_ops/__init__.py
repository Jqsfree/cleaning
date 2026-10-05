"""批次运维工具（合并 / 抽样），跨品类复用。

- ``merge_csvs``  多 CSV 合并去重 → Bronze（``raw/{category}/``）
- ``sample_qc``   抽样 QC（公式算样本量 / SRS / 分层）→ ``02_sample/``

脚本约定见 ``02_脚本/tools/README.md``：纯函数 + ``main()`` 薄 CLI 分层，
纯函数可被 ``tests/`` 直接 import 单测。
"""
