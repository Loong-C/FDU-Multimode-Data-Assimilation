# Git 工作流

在 `codex/` 主题分支上开发。每次提交后，将当前分支正常推送到 `origin` 的同名分支。合并到 `main` 后也推送 `main`。

交付前用 `git ls-remote --heads origin` 核对远端提交号。推送失败时保留本地提交，排查后重试，并说明仍待推送的分支。新克隆也沿用此约定，可显式执行 `git push -u origin HEAD`。
