// Package model contains types and utilities for parsing, validating, and
// working with model names and digests.
package model

import (
	"cmp"
	"encoding/hex"
	"hash/maphash"
	"log/slog"
	"strings"

	"github.com/ollama/ollama/types/structs"
)

// MissingPart is used to indicate any part of a name that was "promised" by
// the presence of a separator, but is missing.
//
// The value was chosen because it is deemed unlikely to be set by a user,
// not a valid part name valid when checked by [Name.IsValid], and easy to
// spot in logs.
const MissingPart = "!MISSING!"

// DefaultName returns a name with the default values for the host, namespace,
// and tag parts. The model and digest parts are empty.
//
//   - The default host is ("registry.ollama.ai")
//   - The default namespace is ("library")
//   - The default tag is ("latest")
func DefaultName() Name {
	return Name{
		Host:      "registry.ollama.ai",
		Namespace: "library",
		Tag:       "latest",
	}
}

type partKind int

const (
	kindHost partKind = iota
	kindNamespace
	kindModel
	kindTag
	kindDigest
)

func (k partKind) String() string {
	switch k {
	case kindHost:
		return "host"
	case kindNamespace:
		return "namespace"
	case kindModel:
		return "model"
	case kindTag:
		return "tag"
	case kindDigest:
		return "digest"
	default:
		return "unknown"
	}
}

// IsValidShort returns true if the namespace and model parts are both valid
// namespace and model parts, respectively.
//
// This can be use to incrementally validate a name as it is being built over
// time.
//
// It is equivalent to:
//
//	(Name{Namespace: namespace, Model: model}).IsValid()
//
// To validate a model or namespace only, use a placeholder for the other,
// such as "x".
func IsValidShort(namespace, model string) bool {
	return isValidPart(kindNamespace, namespace) && isValidPart(kindModel, model)
}

// Name is a structured representation of a model name string, as defined by
// [ParseNameNoDefaults].
//
// It is not guaranteed to be valid. Use [Name.IsValid] to check if the name
// is valid.
//
// It is not directly comparable with other Names. Use [Name.Equal] and
// [Name.MapHash] for determining equality and using as a map key.
type Name struct {
	_ structs.Incomparable

	Host      string
	Namespace string
	Model     string
	Tag       string
	RawDigest string
}

// ParseName returns the result of [ParseNameNoDefaults] with the input
// string merged with [DefaultName].
func ParseName(s string) Name {
	return ParseNameNoDefaults(s).Merge(DefaultName())
}

// ParseNameNoDefaults parses and assembles a Name from a name string. The
// format of a valid name string is:
//
//	  s:
//		  { host } "/" { namespace } "/" { model } ":" { tag } "@" { digest }
//		  { host } "/" { namespace } "/" { model } ":" { tag }
//		  { host } "/" { namespace } "/" { model } "@" { digest }
//		  { host } "/" { namespace } "/" { model }
//		  { host } "/" { model } ":" { tag } "@" { digest }
//		  { host } "/" { model } ":" { tag }
//		  { host } "/" { model } "@" { digest }
//		  { host } "/" { model }
//		  { model } ":" { tag } "@" { digest }
//		  { model } ":" { tag }
//		  { model } "@" { digest }
//		  { model }
//		  "@" { digest }
//	  host:
//	      pattern: alphanum { alphanum | "-" | "_" | "." | ":" }*
//	      length:  [1, 350]
//	  namespace:
//	      pattern: alphanum { alphanum | "-" | "_" }*
//	      length:  [2, 80]
//	  model:
//	      pattern: alphanum { alphanum | "-" | "_" | "." }*
//	      length:  [2, 80]
//	  tag:
//	      pattern: alphanum { alphanum | "-" | "_" | "." }*
//	      length:  [1, 80]
//	  digest:
//	      pattern: alphanum { alphanum | "-" | ":" }*
//	      length:  [2, 80]
//
// Most users should use [ParseName] instead, unless need to support
// different defaults than DefaultName.
//
// The name returned is not guaranteed to be valid. If it is not valid, the
// field values are left in an undefined state. Use [Name.IsValid] to check
// if the name is valid.
func ParseNameNoDefaults(s string) Name {
	var n Name
	var promised bool

	// Digest is the exception to the rule that both parts separated by a
	// separator must be present. If the digest is promised, the digest
	// part must be present, but the name part can be empty/undefined.
	s, n.RawDigest, promised = cutLast(s, "@")
	if promised && n.RawDigest == "" {
		n.RawDigest = MissingPart
	}

	s, n.Tag, _ = cutPromised(s, ":")
	s, n.Model, promised = cutPromised(s, "/")
	if !promised {
		n.Model = s
		return n
	}
	s, n.Namespace, promised = cutPromised(s, "/")
	if !promised {
		n.Namespace = s
		return n
	}
	n.Host = s

	return n
}

// Digest returns the result of [ParseDigest] with the RawDigest field.
func (n Name) Digest() Digest {
	return ParseDigest(n.RawDigest)
}

var mapHashSeed = maphash.MakeSeed()

// MapHash returns a [maphash.Hash] of the name. The hash is suitable for
// use in a map key, and is case-insensitive.
func (n Name) MapHash() uint64 {
	var h maphash.Hash
	h.SetSeed(mapHashSeed)
	writeLower(&h, n.Host)
	writeLower(&h, n.Namespace)
	writeLower(&h, n.Model)
	writeLower(&h, n.Tag)
	writeLower(&h, n.RawDigest)
	return h.Sum64()
}

type writeByter interface {
	WriteByte(byte) error
}

// nolint: errcheck
func writeLower(w writeByter, s string) {
	for i := range s {
		if s[i] >= 'A' && s[i] <= 'Z' {
			w.WriteByte(s[i] - 'A' + 'a')
			continue
		}
		w.WriteByte(s[i])
	}
}

// Equal returns true if n and o are equal case-insensitively.
func (n Name) Equal(o Name) bool {
	return n.MapHash() == o.MapHash()
}

// IsValid returns true all the parts of Name are present, with the
// exception that the digest parts will be considered valid if empty,
// otherwise it must be a valid digest.
func (n Name) IsValid() bool {
	if n.RawDigest != "" && !ParseDigest(n.RawDigest).IsValid() {
		return false
	}
	return n.IsFullyQualified()
}

// IsFullyQualified returns true if all parts of the name are present and
// valid. The digest is ignored.
func (n Name) IsFullyQualified() bool {
	var parts = []string{
		n.Host,
		n.Namespace,
		n.Model,
		n.Tag,
	}
	for i, part := range parts {
		if !isValidPart(partKind(i), part) {
			return false
		}
	}
	return true
}

// String returns the name string, in the format that [ParseNameNoDefaults]
// accepts as valid, if [Name.IsValid] reports true; otherwise the empty
// string is returned.
func (n Name) String() string {
	if !n.IsValid() {
		return ""
	}
	var b strings.Builder
	if n.Host != "" {
		b.WriteString(n.Host)
		b.WriteByte('/')
	}
	if n.Namespace != "" {
		b.WriteString(n.Namespace)
		b.WriteByte('/')
	}
	b.WriteString(n.Model)
	if n.Tag != "" {
		b.WriteByte(':')
		b.WriteString(n.Tag)
	}
	if n.RawDigest != "" {
		b.WriteByte('@')
		b.WriteString(n.RawDigest)
	}
	return b.String()
}

// LogValue returns a slog.Value that represents the name as a string.
func (n Name) LogValue() slog.Value {
	return slog.StringValue(n.String())
}

// Merge sets the host, namespace, and tag parts of n to their counterparts
// in o, if they are empty in n. The model and digest parts are never
// modified.
func (n Name) Merge(o Name) Name {
	n.Host = cmp.Or(n.Host, o.Host)
	n.Namespace = cmp.Or(n.Namespace, o.Namespace)
	n.Tag = cmp.Or(n.Tag, o.Tag)
	return n
}

func isValidLen(kind partKind, s string) bool {
	switch kind {
	case kindHost:
		return len(s) >= 1 && len(s) <= 350
	case kindTag:
		return len(s) >= 1 && len(s) <= 80
	default:
		return len(s) >= 2 && len(s) <= 80
	}
}

func isValidPart(kind partKind, s string) bool {
	if !isValidLen(kind, s) {
		return false
	}
	for i := range s {
		if i == 0 {
			if !isAlphanumeric(s[i]) {
				return false
			}
			continue
		}
		switch s[i] {
		case '_', '-':
		case '.':
			if kind == kindNamespace {
				return false
			}
		case ':':
			if kind != kindHost {
				return false
			}
		default:
			if !isAlphanumeric(s[i]) {
				return false
			}
		}
	}
	return true
}

func isAlphanumeric(c byte) bool {
	return c >= 'A' && c <= 'Z' || c >= 'a' && c <= 'z' || c >= '0' && c <= '9'
}

func cutLast(s, sep string) (before, after string, ok bool) {
	i := strings.LastIndex(s, sep)
	if i >= 0 {
		return s[:i], s[i+len(sep):], true
	}
	return s, "", false
}

// cutPromised cuts the last part of s at the last occurrence of sep. If sep is
// found, the part before and after sep are returned as-is unless empty, in
// which case they are returned as MissingPart, which will cause
// [Name.IsValid] to return false.
func cutPromised(s, sep string) (before, after string, ok bool) {
	before, after, ok = cutLast(s, sep)
	if !ok {
		return before, after, false
	}
	return cmp.Or(before, MissingPart), cmp.Or(after, MissingPart), true
}

type DigestType int

const (
	DigestTypeInvalid DigestType = iota
	DigestTypeSHA256
)

func (t DigestType) String() string {
	if t == DigestTypeSHA256 {
		return "sha256"
	}
	return "unknown"
}

// Digest represents a type and hash of a digest. It is comparable and can
// be used as a map key.
type Digest struct {
	Type DigestType
	Hash [32]byte
}

// ParseDigest parses a digest string into a Digest struct. It accepts both
// the forms:
//
//	sha256:deadbeef
//	sha256-deadbeef
//
// The hash part must be exactly 64 characters long.
//
// The form "type:hash" does not round trip through [Digest.String].
func ParseDigest(s string) Digest {
	typ, hash, ok := cutLast(s, ":")
	if !ok {
		typ, hash, ok = cutLast(s, "-")
		if !ok {
			return Digest{}
		}
	}
	if typ != "sha256" {
		return Digest{}
	}
	var d Digest
	n, err := hex.Decode(d.Hash[:], []byte(hash))
	if err != nil || n != 32 {
		return Digest{}
	}
	return Digest{Type: DigestTypeSHA256, Hash: d.Hash}
}

// IsValid returns true if the digest has a valid Type and Hash.
func (d Digest) IsValid() bool {
	if d.Type != DigestTypeSHA256 {
		return false
	}
	return d.Hash != [32]byte{}
}

// String returns the digest as a string in the form "type-hash". The hash
// is encoded as a hex string.
func (d Digest) String() string {
	var b strings.Builder
	b.WriteString(d.Type.String())
	b.WriteByte('-')
	b.WriteString(hex.EncodeToString(d.Hash[:]))
	return b.String()
}

// LogValue returns a slog.Value that represents the digest as a string.
func (d Digest) LogValue() slog.Value {
	return slog.StringValue(d.String())
}
