package com.sun.corba.se.PortableActivationIDL.RepositoryPackage;

/**
* com/sun/corba/se/PortableActivationIDL/RepositoryPackage/ServerDefHolder.java .
* Generated by the IDL-to-Java compiler (portable), version "3.2"
* from /Users/tester/jenkins/workspace/zulu8-build-macos-aarch64/zulu-src/corba/src/share/classes/com/sun/corba/se/PortableActivationIDL/activation.idl
* Thursday, July 15, 2021 8:28:48 AM PDT
*/

public final class ServerDefHolder implements org.omg.CORBA.portable.Streamable
{
  public com.sun.corba.se.PortableActivationIDL.RepositoryPackage.ServerDef value = null;

  public ServerDefHolder ()
  {
  }

  public ServerDefHolder (com.sun.corba.se.PortableActivationIDL.RepositoryPackage.ServerDef initialValue)
  {
    value = initialValue;
  }

  public void _read (org.omg.CORBA.portable.InputStream i)
  {
    value = com.sun.corba.se.PortableActivationIDL.RepositoryPackage.ServerDefHelper.read (i);
  }

  public void _write (org.omg.CORBA.portable.OutputStream o)
  {
    com.sun.corba.se.PortableActivationIDL.RepositoryPackage.ServerDefHelper.write (o, value);
  }

  public org.omg.CORBA.TypeCode _type ()
  {
    return com.sun.corba.se.PortableActivationIDL.RepositoryPackage.ServerDefHelper.type ();
  }

}
