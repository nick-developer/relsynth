# SPDX-License-Identifier: Apache-2.0
"""Build the air-gapped install bundle of the relsynth service.

    python tools/make_airgap_bundle.py --out dist/airgap            # wheels, chart, docs, checksums
    python tools/make_airgap_bundle.py --out dist/airgap --images   # plus the container images (docker save)

The bundle holds everything an install without internet access needs:

    wheels/            relsynth and every dependency, for Linux x86_64 and the chosen Python
    images/*.tar       relsynth-service (built offline from wheels/) and the images of the
                       compose install (metadata PostgreSQL, Keycloak), with --images
    helm/relsynth-<v>.tgz   the Helm chart (needs helm, or the alpine/helm image)
    compose/           docker-compose.yml, Keycloak realm and .env.example
    docs/              the service documentation
    SHA256SUMS         checksums of every file; INSTALL.md the steps

Transfer it through the organisation's media-import process, then check it with
``sha256sum -c SHA256SUMS`` before use.  docs/service/deployment.md has the steps.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess  # nosec B404 (pip, docker and helm with fixed arguments)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRAS = "service,postgres,mysql,oracle,parquet"
THIRD_PARTY_IMAGES = ("postgres:16-alpine", "keycloak/keycloak:26.0.7")


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)  # noqa: S603  # nosec B603


def version() -> str:
    ns: dict = {}
    exec((ROOT / "src/relsynth/_version.py").read_text(), ns)  # noqa: S102  # nosec B102 (our own version file)
    return ns["__version__"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="dist/airgap")
    ap.add_argument("--python", default="3.12", help="target Python version of the wheels")
    ap.add_argument("--platform", default="manylinux2014_x86_64")
    ap.add_argument("--extras", default=EXTRAS)
    ap.add_argument("--images", action="store_true", help="also build and save the container images")
    a = ap.parse_args(argv)
    v = version()
    out = Path(a.out).resolve()
    shutil.rmtree(out, ignore_errors=True)
    wheels = out / "wheels"
    wheels.mkdir(parents=True)

    run([sys.executable, "-m", "build", "--wheel", "--outdir", str(wheels), str(ROOT)])
    whl = next(wheels.glob(f"relsynth-{v}-*.whl"))
    common = ["--dest", str(wheels), "--python-version", a.python, "--implementation", "cp", "--only-binary=:all:"]
    run([sys.executable, "-m", "pip", "download", *common, "--platform", a.platform,
         "--platform", "manylinux_2_28_x86_64", "--platform", "manylinux_2_17_x86_64", "--platform", "linux_x86_64",
         f"{whl}[{a.extras}]"])
    run([sys.executable, "-m", "pip", "download", *common, "--platform", a.platform, "setuptools>=77", "wheel"])

    for sub in ("compose", "helm", "e2e"):
        src = ROOT / "deploy" / sub
        if src.exists():
            shutil.copytree(src, out / "deploy" / sub, ignore=shutil.ignore_patterns("secrets", ".env", "*.tgz"))
    shutil.copytree(ROOT / "docs" / "service", out / "docs")

    if a.images:
        (out / "images").mkdir()
        wh = ROOT / "deploy" / "docker" / "wheelhouse"
        for f in wheels.glob("*.whl"):
            shutil.copy(f, wh / f.name)
        try:
            tag = f"relsynth-service:{v}"
            run(["docker", "build", "-f", str(ROOT / "deploy/docker/Dockerfile.service"), "--build-arg", "OFFLINE=1",
                 "-t", tag, str(ROOT)])
            run(["docker", "save", "-o", str(out / "images" / f"relsynth-service-{v}.tar"), tag])
        finally:
            for f in wh.glob("*.whl"):
                f.unlink()
        for img in THIRD_PARTY_IMAGES:
            have = subprocess.run(["docker", "image", "inspect", img], capture_output=True)  # noqa: S603, S607  # nosec B603 B607
            if have.returncode != 0:
                run(["docker", "pull", img])
            # one platform only: containerd (k3s, kind, OpenShift nodes) refuses an index whose other blobs are missing
            run(["docker", "save", "--platform", "linux/amd64", "-o",
                 str(out / "images" / (img.replace("/", "_").replace(":", "-") + ".tar")), img])

    chart = ROOT / "deploy" / "helm" / "relsynth"
    (out / "helm").mkdir()
    if shutil.which("helm"):
        run(["helm", "package", str(chart), "-d", str(out / "helm")])
    elif shutil.which("docker"):
        run(["docker", "run", "--rm", "-v", f"{ROOT}:/src:ro", "-v", f"{out / 'helm'}:/out", "alpine/helm:3.16.3",
             "package", "/src/deploy/helm/relsynth", "-d", "/out"])
    else:
        print("helm not found: the chart folder is in deploy/helm instead of a packaged .tgz")

    (out / "INSTALL.md").write_text(INSTALL.format(v=v), encoding="utf-8")
    sums = []
    for f in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        sums.append(f"{h.hexdigest()}  {f.relative_to(out)}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(f"bundle: {out} ({len(sums)} files)")
    return 0


INSTALL = """# relsynth service {v}: air-gapped install

1. Check the bundle: `sha256sum -c SHA256SUMS`.
2. Images: `for f in images/*.tar; do docker load -i "$f"; done`, then push them to the internal
   registry (`docker tag` + `docker push`), or `ctr -n k8s.io images import` on each node.
3. Kubernetes or OpenShift: `helm install relsynth helm/relsynth-*.tgz -n relsynth --create-namespace \\
   --set image.repository=<registry>/relsynth-service --set image.tag={v} -f my-values.yaml`.
4. Small installs: `docker compose -f deploy/compose/docker-compose.yml up -d` (see its header).
5. Python only: `pip install --no-index --find-links wheels "relsynth[service,postgres]"`.

docs/deployment.md covers identity provider, TLS, secrets, backup and upgrades.
"""

if __name__ == "__main__":
    sys.exit(main())
