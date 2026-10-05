"""exo_团队协作 — 团队协作（乒乓球双打 + 网球双打）

品类 id：``exo_团队协作``。子运动：

- ``pingpong/`` — 双人乒乓（原 ``exo_pingpong``）
- ``tennis/`` — 双人网球 / 团队协作宽口径（原 ``exo_tennis``）

批次约定::

    data/runs/exo_团队协作/machine_0922_pingpong/
    data/runs/exo_团队协作/machine_0922_tennis/
    raw/exo_团队协作/{双人乒乓,双人网球}_merged_*.csv

调用::

    from categories.exo_团队协作.pingpong.cleaner import clean as clean_pingpong
    from categories.exo_团队协作.tennis.cleaner import clean as clean_tennis

旧路径 ``categories.exo_pingpong`` / ``categories.exo_tennis`` 保留为兼容 shim。
"""
