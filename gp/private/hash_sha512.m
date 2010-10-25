function h = hash_sha512(x)
%HASH_SHA512  Compute SHA-512 hash
%
%  Description
%    H = HASH_SHA512(X) computes SHA-512 hash for argument X
%    using java.security.MessageDigest class.
%
%  Reference  
%    http://download.oracle.com/javase/1.4.2/docs/api/java/security/MessageDigest.html
%  
  if isempty(x)
    h=[];
  else
    md=java.security.MessageDigest.getInstance('SHA-512');
    md.update(x(:));
    h=md.digest;
  end
