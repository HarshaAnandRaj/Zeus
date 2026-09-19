# LMB5 diagnostic: Git LFS storage and provenance

Storage repair, 2026-09-19. The user requested a fix for the diagnostic exceeding
GitHub's ordinary Git file limit. No experiment, source implementation, numerical
result or frozen scientific verdict was changed.

## Artifact identity

- Path: `zeus_sandbox/universe/reports/lmb5_failure_diagnosis2_20260913.json`
- Size: **110,048,017 bytes**.
- SHA-256: `528d5cde5e0e6ff347057412e0cbcc9ae3dfa503c793030c4af49d0d072671a7`.
- Storage: Git LFS, through the exact-path rule in the root `.gitattributes`.

The complete JSON remains at its original working-tree path. Git stores a small
LFS pointer; the LFS object contains the original bytes. A SHA-256 comparison
verified byte identity after materializing the migrated file. The dependent
interior-state probe's recorded `diagnosis_sha` still matches. Git LFS integrity
checking passes. The outgoing ordinary Git history contains no blob over 100 MiB.

New clones need Git LFS installed. If a checkout contains the pointer rather than
the JSON, fetch and materialize it from the repository root:

```powershell
git lfs install --local
git lfs pull --include="zeus_sandbox/universe/reports/lmb5_failure_diagnosis2_20260913.json"
```

## History boundary and original identifiers

Adding LFS in a new commit alone would leave the oversized blob in the push.
Eight unpublished commits were therefore migrated, starting with the commit that
introduced the diagnostic. All earlier commits were excluded. The published
base `b9e877c7140e2fa869847329e5b5f8238e2b6b0c` and the exclusion boundary
`28ebba1723dfa19474d8c6f9eccae230a69b06d0` remain unchanged. The resulting history
is a descendant of the published base and can use a normal fast-forward push.

For each migrated commit, verification found changes only to `.gitattributes`
and the diagnostic's Git representation. All other file contents, authors and
commit messages were preserved. Original commit identifiers in frozen reports
and historical notes are intentionally retained: use this mapping to locate the
equivalent migrated source snapshot instead of silently rewriting evidence.

| Original unpublished commit | Migrated commit |
|---|---|
| `3f3abdfa359f33e926dcc4aec5a63f9fb95c3626` | `f1cdc0fb7981207bcaf3239367680f689e25b009` |
| `f045902d61a8d80c2ee1ec1e53c6173b344ed49b` | `8197b22f514ccb87455b32c2197106df1653788a` |
| `ff780d57ba77ec045538470d6cb948d67f22dd8d` | `85fb8c97b2a153278cd31f25fe8ef31329631f5e` |
| `218a8386d2cb421eacab14ad4339ded2d4843aaf` | `258390fd66c82e319fae160b19763f2e62660631` |
| `2ff80975ab11dcf2f8eff654d82e3578d137b621` | `aa9baa100fb46d41c0294419a012abc4a2e8e0d9` |
| `cea168e4828ed69be49941e4bba5434a23fba5e7` | `928f186c9ab2988fcc97932996210b73c7af1ee9` |
| `6334537171753721bc8e26c5ea0aa7d7b344a60c` | `f78c5eef1b900e3db472b42f8fa3b2b812ba7d2b` |
| `2e9b4c205f9f37af0c15e8cca05e986aba70e8b4` | `0fca5b7dd23c1b8a9916fd35fed0dde03dccd06b` |

## Local recovery backup

The original tip is protected by the local ref
`refs/backup/lmb5-pre-lfs-20260919`. A verified incremental Git bundle also
preserves the original unpublished history:

- `.git/lmb5-lfs-migration-20260919/before.bundle`
- Bundle SHA-256: `f6182d2ec31e341f3e878200bd37038521d0a8ed51b45977b7ec59d82db68ef3`.
- Required published base: `b9e877c7140e2fa869847329e5b5f8238e2b6b0c`.

The same local directory holds preflight identities, the complete commit mapping
and the migration verification receipt. These backups are local, not part of the
ordinary branch push. The published mapping above preserves the cross-reference
for other clones.
