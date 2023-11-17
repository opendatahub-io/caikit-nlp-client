# Self signed certificates for testing

This is short tutorial for generating / regenerating the certificates use in the unit tests and integration tests of the *caikit-nlp-client* library.

The script to generate the certificates rely upon the [CloudFlare's PKI/TLS toolkit](https://github.com/cloudflare/cfssl#cloudflares-pkitls-toolkit).

To install the toolkit you can follow the instruction in the [Installation page](https://github.com/cloudflare/cfssl#installation). For Macos users running `brew install cfssl`
would probably be more efficient.

To generate the certificates type :

```sh
./gen-certificates.bash
```
