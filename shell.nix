{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    python311
    python311Packages.pip
    python311Packages.requests
    nodejs
    (python311Packages.protobuf.overrideAttrs (oldAttrs: {
      version = "3.20.3";  # Use a version compatible with Modal
    }))
    ffmpeg
    # Add other dependencies your project needs
  ];
}
