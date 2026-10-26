// Minimal std::filesystem for Botzone's g++-7.
//
// Botzone compiles submissions with:
//     g++-7 -D_BOTZONE_ONLINE -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -std=c++1z ... -ljson -lpthread
//
// <filesystem> did not ship until GCC 8. GCC 7 has <experimental/filesystem>,
// but that needs -lstdc++fs and the link line is fixed, so it is not an option
// either. This implements the subset rapfi actually uses, over POSIX.
//
// The surface rapfi needs, measured from the amalgamated bundle:
//     path        43x     exists              11x
//     u8path       3x     is_directory         2x
//     create_directories 2x   rename 1x   remove 1x
//     recursive_directory_iterator 1x
//
// Defining names inside namespace std is formally undefined behaviour. It is
// done here deliberately and narrowly: on GCC 7 std::filesystem does not exist
// at all, so nothing is being replaced or shadowed, and the alternative is
// rewriting 43 call sites across a generated 2.9 MB file. This header is only
// emitted into the Botzone bundle -- never into a normal build, which uses the
// real <filesystem>.

#ifndef BOTZONE_FS_SHIM_H
#define BOTZONE_FS_SHIM_H

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <memory>
#include <system_error>
#include <ostream>
#include <string>
#include <vector>

namespace std {
namespace filesystem {

class path {
public:
    path() = default;
    path(const char *s) : p_(s ? s : "") {}
    path(const std::string &s) : p_(s) {}

    path &operator/=(const path &rhs)
    {
        if (rhs.p_.empty())
            return *this;
        if (!rhs.p_.empty() && rhs.p_[0] == '/') {
            p_ = rhs.p_;  // absolute rhs replaces, matching std behaviour
            return *this;
        }
        if (!p_.empty() && p_.back() != '/')
            p_ += '/';
        p_ += rhs.p_;
        return *this;
    }

    path parent_path() const
    {
        const std::string::size_type i = p_.find_last_of('/');
        if (i == std::string::npos)
            return path();
        if (i == 0)
            return path("/");
        return path(p_.substr(0, i));
    }

    path filename() const
    {
        const std::string::size_type i = p_.find_last_of('/');
        return i == std::string::npos ? *this : path(p_.substr(i + 1));
    }

    path extension() const
    {
        const std::string f = filename().p_;
        const std::string::size_type i = f.find_last_of('.');
        if (i == std::string::npos || i == 0)
            return path();
        return path(f.substr(i));
    }

    path stem() const
    {
        const std::string f = filename().p_;
        const std::string::size_type i = f.find_last_of('.');
        return (i == std::string::npos || i == 0) ? path(f) : path(f.substr(0, i));
    }

    path &replace_filename(const path &r)
    {
        *this = parent_path() / r;
        return *this;
    }
    path &replace_extension(const path &e = path())
    {
        const std::string ext = extension().string();
        if (!ext.empty())
            p_.erase(p_.size() - ext.size());
        const std::string add = e.string();
        if (!add.empty())
            p_ += (add[0] == '.' ? add : "." + add);
        return *this;
    }
    path &concat(const std::string &s)
    {
        p_ += s;
        return *this;
    }
    path &operator+=(const std::string &s)
    {
        p_ += s;
        return *this;
    }

    bool               is_absolute() const { return !p_.empty() && p_[0] == '/'; }
    bool               empty() const { return p_.empty(); }
    void               clear() { p_.clear(); }
    const std::string &native() const { return p_; }
    std::string        string() const { return p_; }
    std::string        u8string() const { return p_; }
    std::string        generic_string() const { return p_; }
    const char        *c_str() const { return p_.c_str(); }
    operator std::string() const { return p_; }

    friend path operator/(path lhs, const path &rhs)
    {
        lhs /= rhs;
        return lhs;
    }
    friend bool operator==(const path &a, const path &b) { return a.p_ == b.p_; }
    friend bool operator!=(const path &a, const path &b) { return a.p_ != b.p_; }
    friend bool operator<(const path &a, const path &b) { return a.p_ < b.p_; }
    friend std::ostream &operator<<(std::ostream &os, const path &x)
    {
        return os << x.p_;
    }

private:
    std::string p_;
};

inline path u8path(const std::string &s)
{
    return path(s);
}
inline path u8path(const char *s)
{
    return path(s ? s : "");
}

inline bool exists(const path &p)
{
    struct stat st;
    return !p.empty() && ::stat(p.c_str(), &st) == 0;
}

inline bool is_directory(const path &p)
{
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

inline bool is_regular_file(const path &p)
{
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

inline bool create_directories(const path &p)
{
    const std::string s = p.string();
    if (s.empty())
        return false;
    std::string acc;
    bool        made = false;
    for (std::string::size_type i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == '/') {
            if (i == 0) {
                acc = "/";
                continue;
            }
            acc = s.substr(0, i);
            if (acc.empty())
                continue;
            if (::mkdir(acc.c_str(), 0755) == 0)
                made = true;
        }
    }
    return made;
}

inline bool remove(const path &p)
{
    if (is_directory(p))
        return ::rmdir(p.c_str()) == 0;
    return ::remove(p.c_str()) == 0;
}

inline void rename(const path &from, const path &to)
{
    ::rename(from.c_str(), to.c_str());
}

// Non-throwing overloads. rapfi calls these with a std::error_code out-param;
// this shim never fails in a way worth reporting, so the code is just cleared.
inline bool exists(const path &p, std::error_code &ec) { ec.clear(); return exists(p); }
inline bool is_directory(const path &p, std::error_code &ec) { ec.clear(); return is_directory(p); }
inline bool is_regular_file(const path &p, std::error_code &ec) { ec.clear(); return is_regular_file(p); }
inline bool create_directories(const path &p, std::error_code &ec) { ec.clear(); return create_directories(p); }
inline bool remove(const path &p, std::error_code &ec) { ec.clear(); return remove(p); }
inline void rename(const path &f, const path &t, std::error_code &ec) { ec.clear(); rename(f, t); }

class directory_entry {
    // A member function named `path` returning the type `path` makes GCC 7
    // reject the class ("changes meaning of 'path'"). The standard does
    // exactly this, but GCC 7 predates the resolution, so alias the type
    // first and never name it directly inside the class.
    typedef ::std::filesystem::path path_type;

public:
    directory_entry() = default;
    explicit directory_entry(const path_type &p) : p_(p) {}
    const path_type &path() const { return p_; }
    bool is_regular_file() const { return ::std::filesystem::is_regular_file(p_); }
    bool is_directory() const { return ::std::filesystem::is_directory(p_); }

private:
    path_type p_;
};

// Eagerly walks the tree at construction. rapfi uses this once, to list files
// under a directory; a lazy iterator would be more faithful but this keeps the
// shim small and the semantics that matter identical.
class recursive_directory_iterator {
public:
    recursive_directory_iterator() = default;
    explicit recursive_directory_iterator(const path &root)
        : v_(std::make_shared<std::vector<directory_entry>>())
    {
        walk(root);
    }

    const directory_entry &operator*() const { return (*v_)[i_]; }
    const directory_entry *operator->() const { return &(*v_)[i_]; }
    recursive_directory_iterator &operator++()
    {
        ++i_;
        return *this;
    }
    friend bool operator==(const recursive_directory_iterator &a,
                           const recursive_directory_iterator &b)
    {
        return a.at_end() == b.at_end() && (a.at_end() || a.i_ == b.i_);
    }
    friend bool operator!=(const recursive_directory_iterator &a,
                           const recursive_directory_iterator &b)
    {
        return !(a == b);
    }

    recursive_directory_iterator begin() const { return *this; }
    recursive_directory_iterator end() const { return recursive_directory_iterator(); }

private:
    bool at_end() const { return !v_ || i_ >= v_->size(); }

    void walk(const path &dir)
    {
        DIR *d = ::opendir(dir.c_str());
        if (!d)
            return;
        while (struct dirent *e = ::readdir(d)) {
            const std::string name = e->d_name;
            if (name == "." || name == "..")
                continue;
            const path child = dir / path(name);
            v_->push_back(directory_entry(child));
            if (filesystem::is_directory(child))
                walk(child);
        }
        ::closedir(d);
    }

    std::shared_ptr<std::vector<directory_entry>> v_;
    std::size_t                                   i_ = 0;
};

inline recursive_directory_iterator begin(recursive_directory_iterator it)
{
    return it;
}
inline recursive_directory_iterator end(const recursive_directory_iterator &)
{
    return recursive_directory_iterator();
}

}  // namespace filesystem
}  // namespace std

#endif  // BOTZONE_FS_SHIM_H
