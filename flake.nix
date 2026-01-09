{
  description = "Python devShell flake";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.python312
          pkgs.uv
          pkgs.redis
        ];
        LD_LIBRARY_PATH = "/run/opengl-driver/lib";
      };
    };
}
