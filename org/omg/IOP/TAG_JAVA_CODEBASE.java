package org.omg.IOP;


/**
* org/omg/IOP/TAG_JAVA_CODEBASE.java .
* Generated by the IDL-to-Java compiler (portable), version "3.2"
* from /Users/tester/jenkins/workspace/zulu8-build-macos-aarch64/zulu-src/corba/src/share/classes/org/omg/PortableInterceptor/IOP.idl
* Thursday, July 15, 2021 8:28:48 AM PDT
*/

public interface TAG_JAVA_CODEBASE
{

  /**
       * Class downloading is supported for stubs, ties, values, and 
       * value helpers. The specification allows transmission of codebase 
       * information on the wire for stubs and ties, and enables usage of 
       * pre-existing ClassLoaders when relevant.  
       * <p>
       * For values and value helpers, the codebase is transmitted after the 
       * value tag.  For stubs and ties, the codebase is transmitted as 
       * the TaggedComponent <code>TAG_JAVA_CODEBASE</code> in the IOR 
       * profile, where the <code>component_data</code> is a CDR encapsulation 
       * of the codebase written as an IDL string. The codebase is a 
       * space-separated list of one or more URLs.
       */
  public static final int value = (int)(25L);
}
